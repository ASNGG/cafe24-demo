"""
api/routes_guardian.py - Data Guardian 보안 감시 API
(룰엔진 + IsolationForest ML + LangChain Agent)
"""
import os
import sqlite3
import time as _time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import joblib
from fastapi import APIRouter, Depends, Request, HTTPException

from core.utils import safe_str
import state as st
from api.common import verify_credentials, error_response


router = APIRouter(prefix="/api", tags=["guardian"])

# ─────────────────────────────────────────────────────────
# 공통 상수
# ─────────────────────────────────────────────────────────
_ACTION_MAP = {"INSERT": 0, "SELECT": 0, "UPDATE": 1, "DELETE": 2,
               "ALTER": 3, "DROP": 4, "TRUNCATE": 4}

# ─────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────
_GUARDIAN_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "guardian.db")


def _guardian_conn():
    conn = sqlite3.connect(_GUARDIAN_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _guardian_db():
    """Guardian DB 연결 컨텍스트 매니저 (자동 commit/close)"""
    conn = _guardian_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _guardian_init():
    """감사 로그 테이블 + 시드 데이터 초기화"""
    with _guardian_db() as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            affected_amount REAL DEFAULT 0,
            status TEXT DEFAULT 'executed',
            risk_level TEXT DEFAULT 'LOW',
            agent_reason TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_count INTEGER,
            was_mistake INTEGER DEFAULT 0,
            description TEXT
        )""")
        # 시드 데이터 (없을 때만)
        if c.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0:
            import random
            from datetime import timedelta
            _now = datetime.now()
            _users = ["kim", "park", "lee", "choi", "jung"]
            _tables = ["orders", "payments", "users", "products", "shipments", "logs", "temp_reports"]
            for _ in range(200):
                ts = (_now - timedelta(days=random.randint(0, 30), hours=random.randint(0, 12))).isoformat()
                u = random.choice(_users)
                act = random.choice(["INSERT", "UPDATE", "DELETE"])
                tbl = random.choice(_tables)
                rc = random.randint(1, 8) if act == "DELETE" else random.randint(1, 20)
                amt = rc * random.randint(30000, 120000) if tbl in ("orders", "payments") else 0
                c.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) VALUES (?,?,?,?,?,?,?,?)",
                          (ts, u, act, tbl, rc, amt, "executed", "LOW"))
            _incidents = [
                ("DELETE", "orders", 250, 1, "신입 직원이 WHERE 없이 DELETE 실행, 전체 복구"),
                ("DELETE", "orders", 180, 1, "테스트 DB와 혼동하여 프로덕션에서 삭제"),
                ("DELETE", "payments", 320, 1, "정산 데이터 삭제 실수, DBA가 백업에서 복구"),
                ("DELETE", "orders", 150, 1, "퇴근 전 급하게 작업하다 실수"),
                ("DELETE", "users", 500, 1, "탈퇴 처리 스크립트 오류로 활성 유저 삭제"),
                ("DELETE", "products", 200, 1, "카테고리 정리 중 실수로 전체 삭제"),
                ("DELETE", "orders", 400, 1, "연말 정산 중 데이터 혼동"),
                ("DELETE", "logs", 10000, 0, "정기 로그 정리 (스케줄 작업)"),
                ("DELETE", "temp_reports", 5000, 0, "임시 리포트 정리"),
                ("UPDATE", "orders", 300, 1, "금액 필드 일괄 0으로 업데이트 실수"),
                ("UPDATE", "products", 150, 1, "가격 일괄 변경 시 WHERE 조건 누락"),
                ("UPDATE", "users", 1000, 1, "권한 일괄 변경 실수"),
            ]
            for act, tbl, rc, mis, desc in _incidents:
                c.execute("INSERT INTO incidents (action,table_name,row_count,was_mistake,description) VALUES (?,?,?,?,?)",
                          (act, tbl, rc, mis, desc))


# ── 룰엔진 ──
_CORE_TABLES = {"orders", "payments", "users", "products", "shipments"}


def _rule_engine_evaluate(action: str, table: str, row_count: int, hour: int = None):
    """룰 기반 1차 필터 (<1ms)"""
    start = _time.perf_counter()
    reasons = []
    level = "pass"
    if action in ("DROP", "TRUNCATE", "ALTER"):
        reasons.append(f"DDL 명령어 ({action}) 감지")
        level = "block"
    if action in ("DELETE", "UPDATE") and table in _CORE_TABLES and row_count > 100:
        reasons.append(f"핵심 테이블({table}) 대량 {action} ({row_count}건)")
        level = "block"
    elif action == "DELETE" and row_count > 1000:
        reasons.append(f"대량 삭제 ({row_count}건)")
        level = "block"
    elif action == "DELETE" and table in _CORE_TABLES and row_count > 10:
        reasons.append(f"핵심 테이블({table}) 삭제 {row_count}건")
        if level != "block":
            level = "warn"
    if hour is not None and (hour >= 22 or hour < 6):
        if table in _CORE_TABLES and action in ("DELETE", "UPDATE"):
            reasons.append(f"업무 시간 외 ({hour}시) 핵심 데이터 수정")
            if level == "pass":
                level = "warn"
            elif level == "warn":
                level = "block"
    elapsed = (_time.perf_counter() - start) * 1000
    if not reasons:
        reasons.append("정상 범위 쿼리")
    return {"level": level, "reasons": reasons, "elapsed_ms": round(elapsed, 3)}


# ── ML 이상탐지 (Isolation Forest) ──
_GUARDIAN_ISO_MODEL = None
_GUARDIAN_SCALER = None


def _guardian_train_model():
    """pkl 파일 우선 로드, 없으면 감사 로그 기반 인라인 학습"""
    global _GUARDIAN_ISO_MODEL, _GUARDIAN_SCALER
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # (1) pkl 파일 로드 시도
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_guardian_anomaly.pkl")
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaler_guardian.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        _GUARDIAN_ISO_MODEL = joblib.load(model_path)
        _GUARDIAN_SCALER = joblib.load(scaler_path)
        st.logger.info("Guardian ML: pkl 파일에서 모델 로드 완료")
        return

    # (2) pkl 없으면 DB에서 인라인 학습
    with _guardian_db() as conn:
        rows = conn.execute(
            "SELECT user_id, action, table_name, row_count, affected_amount, timestamp "
            "FROM audit_log WHERE status='executed'"
        ).fetchall()

    if len(rows) < 20:
        st.logger.warning("Guardian ML: 학습 데이터 부족 (%d건) — pkl 파일도 없음", len(rows))
        return

    features = []
    for r in rows:
        ts = r["timestamp"] or ""
        hour = int(ts[11:13]) if len(ts) > 13 else 12
        features.append([
            _ACTION_MAP.get(r["action"], 0),
            1 if r["table_name"] in _CORE_TABLES else 0,
            r["row_count"],
            np.log1p(r["row_count"]),
            r["affected_amount"],
            hour,
            1 if (hour >= 22 or hour < 6) else 0,
        ])

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    _GUARDIAN_ISO_MODEL = model
    _GUARDIAN_SCALER = scaler
    st.logger.info("Guardian ML: IsolationForest 인라인 학습 완료 (%d건, 7 features)", len(rows))


def _guardian_anomaly_score(user_id: str, action: str, table: str, row_count: int, hour: int = None):
    """단일 쿼리의 이상 점수 반환 (0~1, 높을수록 이상)"""
    global _GUARDIAN_ISO_MODEL, _GUARDIAN_SCALER
    h = hour if hour is not None else 12

    if _GUARDIAN_ISO_MODEL is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_guardian_anomaly.pkl")
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaler_guardian.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                _GUARDIAN_ISO_MODEL = joblib.load(model_path)
                _GUARDIAN_SCALER = joblib.load(scaler_path)
                st.logger.info("Guardian ML: lazy loading으로 pkl 모델 로드 완료")
            except Exception as e:
                st.logger.error("Guardian ML: lazy loading 실패 — %s", e)
                return None
        else:
            return None

    avg_amounts = {"orders": 67500, "payments": 67500, "products": 35000}
    amount = row_count * avg_amounts.get(table, 0)

    feat = np.array([[
        _ACTION_MAP.get(action, 0),
        1 if table in _CORE_TABLES else 0,
        row_count,
        np.log1p(row_count),
        amount,
        h,
        1 if (h >= 22 or h < 6) else 0,
    ]])
    feat_scaled = _GUARDIAN_SCALER.transform(feat)
    raw_score = -_GUARDIAN_ISO_MODEL.decision_function(feat_scaled)[0]
    score = min(max((raw_score + 0.5) / 1.0, 0.0), 1.0)

    # 사용자별 이탈도
    with _guardian_db() as conn:
        user_avg = conn.execute(
            "SELECT AVG(row_count) as avg_rc FROM audit_log "
            "WHERE user_id=? AND action=? AND status='executed'",
            (user_id, action)
        ).fetchone()

    user_deviation = 0.0
    if user_avg and user_avg["avg_rc"] and user_avg["avg_rc"] > 0:
        user_deviation = min(row_count / user_avg["avg_rc"], 10.0) / 10.0

    combined = score * 0.6 + user_deviation * 0.4

    # SHAP 기반 위험 요인 분석
    FEATURE_LABELS = {
        0: ("작업 위험도", lambda v: {"0": "SELECT/INSERT", "1": "UPDATE", "2": "DELETE", "3": "ALTER", "4": "DROP/TRUNCATE"}.get(str(int(v)), action) + " 작업"),
        1: ("핵심 테이블", lambda v: f"{table}은 핵심 테이블" if v == 1 else "비핵심 테이블"),
        2: ("대상 행 수", lambda v: f"{int(v):,}건"),
        3: ("행 수(log)", None),
        4: ("추정 금액", lambda v: f"\u20a9{v:,.0f}"),
        5: ("시간대", lambda v: f"{int(v)}시"),
        6: ("야간 여부", lambda v: f"야간 작업 ({int(feat[0][5])}시)" if v == 1 else "주간 작업"),
    }
    risk_factors = []
    try:
        import shap
        explainer = shap.TreeExplainer(_GUARDIAN_ISO_MODEL)
        shap_values = explainer.shap_values(feat_scaled)
        sv = shap_values[0]
        for i in sorted(range(len(sv)), key=lambda x: sv[x]):
            if i == 3:
                continue
            label, detail_fn = FEATURE_LABELS[i]
            if detail_fn is None:
                continue
            contribution = -sv[i]
            if contribution > 0.01:
                severity = "high" if contribution > 0.05 else "medium"
                risk_factors.append({
                    "factor": label,
                    "detail": detail_fn(feat[0][i]),
                    "severity": severity,
                    "contribution": round(float(contribution), 4),
                })
    except Exception:
        z_scores = feat_scaled[0]
        for i, z in enumerate(z_scores):
            if i == 3 or abs(z) <= 0.8:
                continue
            label, detail_fn = FEATURE_LABELS[i]
            if detail_fn is None:
                continue
            risk_factors.append({
                "factor": label,
                "detail": detail_fn(feat[0][i]),
                "severity": "high" if abs(z) > 2 else "medium",
                "contribution": round(abs(float(z)) / 10, 4),
            })

    return {
        "anomaly_score": round(score, 4),
        "user_deviation": round(user_deviation, 4),
        "combined_score": round(combined, 4),
        "model": "IsolationForest",
        "features_used": 7,
        "risk_factors": risk_factors,
    }


# ── ML 결과 LLM 해석 ──

def _guardian_ml_interpret(ml_result: dict, action: str, table: str, row_count: int) -> str:
    """ML 이상탐지 결과를 LLM이 한 줄로 해석"""
    api_key = st.OPENAI_API_KEY
    if not api_key or not ml_result:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key, max_tokens=150)

        score = ml_result["combined_score"]
        risk_text = ""
        if ml_result.get("risk_factors"):
            factors = [f"{rf['factor']}({rf['detail']})" for rf in ml_result["risk_factors"][:3]]
            risk_text = f"주요 위험 요인: {', '.join(factors)}"

        prompt = (
            f"DB 감사 시스템의 ML 이상탐지 결과를 간결하게 1~2문장으로 해석해줘.\n"
            f"- 작업: {action} on {table} ({row_count:,}건)\n"
            f"- 종합 이상 점수: {score*100:.1f}/100\n"
            f"- 이상 점수(모델): {ml_result['anomaly_score']*100:.1f}%\n"
            f"- 사용자 이탈도: {ml_result['user_deviation']*100:.1f}%\n"
            f"{risk_text}\n"
            f"점수 기준: 0~40 정상, 40~70 주의, 70+ 위험. 한국어로 핵심만 답변."
        )
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        st.logger.warning("Guardian ML interpret error: %s", e)
        return None


# ── Guardian Agent (LangChain create_agent) ──

def _guardian_tool_defs(conn, outer_row_count: int):
    """Guardian Agent용 Tool 함수들을 정의하고 반환"""

    def analyze_impact(table_name: str, row_count: int) -> str:
        """삭제/수정 대상 데이터의 비즈니스 영향도를 분석한다."""
        avg_amounts = {"orders": 67500, "payments": 67500, "users": 0, "products": 35000, "shipments": 15000}
        related = {"orders": ["order_items", "payments", "shipments"], "payments": ["refunds", "settlements"],
                   "users": ["orders", "reviews", "addresses"], "products": ["order_items", "reviews", "inventory"],
                   "shipments": ["tracking_logs"]}
        amount = row_count * avg_amounts.get(table_name, 0)
        rel = related.get(table_name, [])
        diff = "높음" if table_name in ("orders", "payments") else "중간"
        return f"영향: {table_name} {row_count}건, 추정 금액: \u20a9{amount:,.0f}, 연쇄 테이블: {', '.join(rel) if rel else '없음'} ({len(rel)}개), 복구 난이도: {diff}"

    def get_user_pattern(user_id: str, current_row_count: int) -> str:
        """해당 사용자의 최근 30일 행동 패턴을 조회한다."""
        row = conn.cursor().execute("SELECT AVG(row_count) as avg_c, COUNT(*) as cnt, MAX(row_count) as mx FROM audit_log WHERE user_id=? AND action IN ('DELETE','UPDATE') AND timestamp > datetime('now','-30 days')", (user_id,)).fetchone()
        if not row or not row["avg_c"]:
            return f"'{user_id}'의 최근 30일 이력 없음. 첫 작업."
        avg = row["avg_c"]
        dev = current_row_count / avg if avg > 0 else 999
        res = f"'{user_id}' 패턴: 평균 {avg:.0f}건/회, 최대 {row['mx']}건, 총 {row['cnt']}회. 현재 {current_row_count}건 = 평소의 {dev:.1f}배"
        if dev > 10:
            res += " - 극단적 이탈"
        elif dev > 3:
            res += " - 유의미한 이탈"
        return res

    def search_similar(action: str, table_name: str) -> str:
        """과거 유사 사건을 검색한다."""
        rows = conn.cursor().execute("SELECT * FROM incidents WHERE action=? AND row_count >= 50 ORDER BY ABS(row_count - ?) LIMIT 10", (action, outer_row_count)).fetchall()
        if not rows:
            return "유사 사례 없음"
        mistakes = sum(1 for r in rows if r["was_mistake"])
        total = len(rows)
        details = [f"  - [{'실수' if r['was_mistake'] else '정상'}] {r['table_name']} {r['row_count']}건: {r['description']}" for r in rows[:5]]
        return f"유사 {total}건 중 {mistakes}건 실수 ({mistakes/total*100:.0f}%)\n" + "\n".join(details)

    def execute_decision(decision: str, reason: str) -> str:
        """차단 또는 승인 결정을 실행한다."""
        return f"{'차단' if decision == 'block' else '승인'} 실행 완료. 사유: {reason}"

    return [analyze_impact, get_user_pattern, search_similar, execute_decision]


def _recovery_tool_defs(conn):
    """Recovery Agent용 Tool 함수들"""

    def search_audit_log(user_id: str = "", table_name: str = "", action: str = "DELETE") -> str:
        """감사 로그에서 최근 차단/삭제된 기록을 검색한다."""
        query = "SELECT * FROM audit_log WHERE status IN ('blocked','executed') AND action=?"
        params = [action]
        if user_id:
            query += " AND user_id=?"
            params.append(user_id)
        if table_name:
            query += " AND table_name=?"
            params.append(table_name)
        query += " ORDER BY id DESC LIMIT 5"
        rows = conn.cursor().execute(query, params).fetchall()
        if not rows:
            return "해당 조건의 감사 로그 없음"
        return "\n".join([f"  - [{r['id']}] {r['timestamp'][:16]} {r['user_id']}: {r['action']} {r['table_name']} {r['row_count']}건 (\u20a9{r['affected_amount']:,.0f}) [{r['status']}]" for r in rows])

    def generate_restore_sql(table_name: str, row_count: int, description: str) -> str:
        """복구 SQL을 생성한다. 직접 실행하지 않고 DBA 승인 필요."""
        return f"""복구 SQL 생성 완료 (DBA 승인 필요):

-- 대상: {table_name} {row_count}건
-- 설명: {description}

INSERT INTO {table_name}
SELECT * FROM {table_name}_audit_backup
WHERE deleted_at >= NOW() - INTERVAL 1 HOUR;

-- 정합성 검증
SELECT COUNT(*) as restored FROM {table_name}
WHERE created_at >= NOW() - INTERVAL 1 HOUR;"""

    return [search_audit_log, generate_restore_sql]


def _extract_agent_steps(messages):
    """create_agent 결과 메시지에서 tool 호출 내역 추출"""
    steps = []
    from langchain_core.messages import AIMessage, ToolMessage
    tool_calls_map = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_map[tc["id"]] = {"tool": tc["name"], "input": str(tc["args"])}
        elif isinstance(msg, ToolMessage):
            tc_id = msg.tool_call_id
            if tc_id in tool_calls_map:
                tool_calls_map[tc_id]["output"] = msg.content
                steps.append(tool_calls_map[tc_id])
    return steps


def _run_guardian_agent(user_id: str, action: str, table: str, row_count: int, api_key: str):
    """LangChain create_agent로 위험도 상세 분석"""
    try:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        return {"output": f"LangChain 미설치: {e}", "steps": []}

    with _guardian_db() as conn:
        tools = _guardian_tool_defs(conn, row_count)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""당신은 Data Guardian Agent입니다. DB 변경 요청의 위험도를 분석합니다.

순서: 1) analyze_impact로 영향도 분석 2) get_user_pattern으로 사용자 패턴 확인 3) search_similar로 유사 사례 검색 4) execute_decision으로 최종 판단

최종 응답에 포함: 영향 범위(건수/금액), 위험 사유, 유사 사례 통계, 권고. 한국어로 응답.""",
        )

        result = graph.invoke({"messages": [{"role": "user", "content": f"사용자: {user_id}, 작업: {action}, 테이블: {table}, 대상: {row_count}건. 위험도 분석 후 차단 여부 판단해주세요."}]})

    messages = result.get("messages", [])
    steps = _extract_agent_steps(messages)
    final_output = messages[-1].content if messages else "분석 완료"
    return {"output": final_output, "steps": steps}


def _run_recovery_agent(message: str, api_key: str):
    """복구 요청 처리 Agent"""
    try:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        return {"output": f"LangChain 미설치: {e}", "steps": []}

    with _guardian_db() as conn:
        tools = _recovery_tool_defs(conn)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""당신은 Data Guardian 복구 Agent입니다.
사용자의 자연어 복구 요청을 받아서:
1) search_audit_log로 관련 기록 검색
2) generate_restore_sql로 복구 SQL 생성
복구 SQL은 직접 실행하지 않고 DBA 승인을 받아야 합니다. 한국어로 응답.""",
        )

        result = graph.invoke({"messages": [{"role": "user", "content": message}]})

    messages = result.get("messages", [])
    steps = _extract_agent_steps(messages)
    final_output = messages[-1].content if messages else "복구 계획 수립 완료"
    return {"output": final_output, "steps": steps}


# ── Lazy 초기화 ──
_GUARDIAN_INITIALIZED = False


def _ensure_guardian_init():
    """Guardian DB + ML 모델 lazy 초기화"""
    global _GUARDIAN_INITIALIZED
    if _GUARDIAN_INITIALIZED:
        return
    try:
        _guardian_init()
        _guardian_train_model()
        _GUARDIAN_INITIALIZED = True
    except Exception as e:
        st.logger.warning("Guardian init failed: %s", e)


# ── Endpoints ──

@router.post("/guardian/analyze")
async def guardian_analyze(request: Request, user: dict = Depends(verify_credentials)):
    """Data Guardian: 쿼리 위험도 분석 (룰엔진 + ML 이상탐지 + AI Agent)"""
    _ensure_guardian_init()
    body = await request.json()
    user_id = body.get("user_id", "unknown")
    action = body.get("action", "DELETE")
    table = body.get("table", "orders")
    row_count = int(body.get("row_count", 1))
    hour = body.get("hour")
    mode = body.get("mode", "rule+ml")

    # 1) 룰엔진
    rule = None
    if mode in ("rule", "rule+ml"):
        rule = _rule_engine_evaluate(action, table, row_count, hour)

    # 2) ML 이상탐지
    ml = None
    if mode in ("ml", "rule+ml"):
        ml = _guardian_anomaly_score(user_id, action, table, row_count, hour)

    # ML 해석 (LLM)
    if ml:
        interpretation = _guardian_ml_interpret(ml, action, table, row_count)
        if interpretation:
            ml["interpretation"] = interpretation

    # 판단 로직
    need_agent = False
    effective_level = "pass"

    if mode == "rule" and rule:
        effective_level = rule["level"]
        need_agent = (rule["level"] == "block")
    elif mode == "ml":
        if ml is None:
            return {"status": "success",
                    "rule": None, "ml": None, "agent": None,
                    "ml_not_ready": True,
                    "message": "ML 모델이 아직 학습되지 않았습니다. 감사 로그가 20건 이상 쌓이면 자동으로 학습됩니다."}
        if ml["combined_score"] > 0.7:
            effective_level = "block"
            need_agent = True
        elif ml["combined_score"] > 0.4:
            effective_level = "warn"
        else:
            effective_level = "pass"
    elif mode == "rule+ml":
        rule_level = rule["level"] if rule else "pass"
        ml_level = "pass"
        if ml:
            if ml["combined_score"] > 0.7:
                ml_level = "block"
            elif ml["combined_score"] > 0.4:
                ml_level = "warn"
        level_order = {"pass": 0, "warn": 1, "block": 2}
        effective_level = rule_level if level_order.get(rule_level, 0) >= level_order.get(ml_level, 0) else ml_level
        need_agent = (effective_level == "block")

    # 통과/경고 -> 로그만 기록
    if effective_level == "pass":
        with _guardian_db() as conn:
            conn.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) VALUES (?,?,?,?,?,?,?,?)",
                         (datetime.now().isoformat(), user_id, action, table, row_count, 0, "executed", "LOW"))
        return {"status": "success", "rule": rule, "ml": ml, "agent": None}

    if effective_level == "warn":
        with _guardian_db() as conn:
            conn.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) VALUES (?,?,?,?,?,?,?,?)",
                         (datetime.now().isoformat(), user_id, action, table, row_count, 0, "warned", "MEDIUM"))
        return {"status": "success", "rule": rule, "ml": ml, "agent": None}

    # 차단 -> Agent 호출
    api_key = st.OPENAI_API_KEY
    if not api_key:
        return {"status": "success", "rule": rule, "ml": ml, "agent": {"output": "OpenAI API Key 미설정. Agent 분석 불가.", "steps": []}}

    try:
        agent_result = _run_guardian_agent(user_id, action, table, row_count, api_key)
    except Exception as e:
        st.logger.error("Guardian agent error: %s", e)
        agent_result = {"output": f"Agent 오류: {str(e)}", "steps": []}

    # 차단 로그 기록
    avg_amounts = {"orders": 67500, "payments": 67500, "products": 35000}
    amount = row_count * avg_amounts.get(table, 0)
    with _guardian_db() as conn:
        conn.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level,agent_reason) VALUES (?,?,?,?,?,?,?,?,?)",
                     (datetime.now().isoformat(), user_id, action, table, row_count, amount, "blocked", "HIGH", agent_result["output"][:300]))

    return {"status": "success", "rule": rule, "ml": ml, "agent": agent_result}


@router.post("/guardian/recover")
async def guardian_recover(request: Request, user: dict = Depends(verify_credentials)):
    """Data Guardian: 자연어 복구 요청"""
    body = await request.json()
    message = body.get("message", "")
    if not message:
        raise HTTPException(400, "message 필수")

    api_key = st.OPENAI_API_KEY
    if not api_key:
        return error_response("OpenAI API Key 미설정")

    try:
        result = _run_recovery_agent(message, api_key)
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("Guardian recovery error: %s", e)
        return error_response(str(e))


@router.get("/guardian/logs")
async def guardian_logs(user: dict = Depends(verify_credentials), limit: int = 30, status_filter: str = ""):
    """Data Guardian: 감사 로그 조회"""
    _ensure_guardian_init()
    with _guardian_db() as conn:
        if status_filter:
            rows = conn.execute("SELECT * FROM audit_log WHERE status=? ORDER BY id DESC LIMIT ?", (status_filter, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return {"status": "success", "logs": [dict(r) for r in rows]}


@router.get("/guardian/stats")
async def guardian_stats(user: dict = Depends(verify_credentials)):
    """Data Guardian: 통계"""
    _ensure_guardian_init()
    with _guardian_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status='blocked'").fetchone()[0]
        warned = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status='warned'").fetchone()[0]
        restored = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status='restored'").fetchone()[0]
        saved = conn.execute("SELECT COALESCE(SUM(affected_amount),0) FROM audit_log WHERE status='blocked'").fetchone()[0]
        daily = conn.execute("SELECT date(timestamp) as d, COUNT(*) as cnt FROM audit_log WHERE status='blocked' AND timestamp > datetime('now','-7 days') GROUP BY d ORDER BY d").fetchall()
    return {
        "status": "success",
        "total": total, "blocked": blocked, "warned": warned, "restored": restored,
        "saved_amount": saved,
        "daily_blocked": [{"date": r["d"], "count": r["cnt"]} for r in daily],
    }


@router.post("/guardian/notify-dba")
async def guardian_notify_dba(request: Request, user: dict = Depends(verify_credentials)):
    """Data Guardian: DBA에게 Resend를 통해 이메일 알림 발송"""
    body = await request.json()
    dba_email = body.get("email", "")
    alert_data = body.get("alert", {})

    if not dba_email:
        raise HTTPException(400, "email 필수")

    resend_key = os.environ.get("RESEND_API_KEY", "")

    user_id = alert_data.get("user_id", "unknown")
    action = alert_data.get("action", "")
    table = alert_data.get("table", "")
    row_count = alert_data.get("row_count", 0)
    rule_reasons = alert_data.get("rule_reasons", [])
    agent_output = alert_data.get("agent_output", "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reasons_html = "".join(f"<li>{r}</li>" for r in rule_reasons) if rule_reasons else "<li>고위험 쿼리 감지</li>"
    agent_html = (agent_output or "Agent 분석 없음").replace("\n", "<br>")

    html = f"""<div style="font-family:'Apple SD Gothic Neo','Malgun Gothic',sans-serif;max-width:640px;margin:0 auto;background:#fff;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden">
  <div style="background:linear-gradient(135deg,#dc2626,#ef4444);padding:24px 32px">
    <h1 style="color:#fff;font-size:20px;margin:0;font-weight:700">Data Guardian Alert</h1>
    <p style="color:#fecaca;font-size:13px;margin:6px 0 0">고위험 쿼리 -- DBA 승인 필요</p>
  </div>
  <div style="padding:28px 32px">
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:20px">
      <tr><td style="padding:8px 12px;background:#fef2f2;font-weight:600;width:100px;border:1px solid #fecaca">시간</td><td style="padding:8px 12px;border:1px solid #fecaca">{ts}</td></tr>
      <tr><td style="padding:8px 12px;background:#fef2f2;font-weight:600;border:1px solid #fecaca">사용자</td><td style="padding:8px 12px;border:1px solid #fecaca">{user_id}</td></tr>
      <tr><td style="padding:8px 12px;background:#fef2f2;font-weight:600;border:1px solid #fecaca">쿼리</td><td style="padding:8px 12px;border:1px solid #fecaca"><code>{action} FROM {table}</code> ({row_count}건)</td></tr>
    </table>
    <div style="margin-bottom:20px">
      <p style="color:#dc2626;font-size:13px;font-weight:600;margin:0 0 8px">차단 사유</p>
      <ul style="margin:0;padding-left:20px;font-size:13px;color:#374151;line-height:1.8">{reasons_html}</ul>
    </div>
    <div style="margin-bottom:20px">
      <p style="color:#4f46e5;font-size:13px;font-weight:600;margin:0 0 8px">AI Agent 분석</p>
      <div style="background:#f8fafc;padding:14px 18px;border-radius:8px;border-left:3px solid #4f46e5;font-size:13px;color:#374151;line-height:1.7">{agent_html}</div>
    </div>
    <div style="text-align:center;padding:16px 0">
      <p style="font-size:14px;color:#6b7280;margin:0 0 12px">이 쿼리를 승인하시겠습니까?</p>
      <a href="#" style="display:inline-block;background:#dc2626;color:#fff;padding:10px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;margin-right:8px">차단 유지</a>
      <a href="#" style="display:inline-block;background:#f59e0b;color:#fff;padding:10px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px">승인</a>
    </div>
  </div>
  <div style="background:#f9fafb;padding:16px 32px;border-top:1px solid #e5e7eb">
    <p style="margin:0;font-size:11px;color:#9ca3af">CAFE24 Data Guardian -- 자동 발송 알림</p>
  </div>
</div>"""

    if resend_key:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                    json={
                        "from": "CAFE24 Guardian <onboarding@resend.dev>",
                        "to": [dba_email],
                        "subject": f"[Guardian Alert] {action} {table} {row_count}건 -- DBA 승인 필요",
                        "html": html,
                    },
                )
            st.logger.info("[guardian-notify] resend status=%s", resp.status_code)
            if resp.status_code >= 400:
                return error_response(f"Resend 오류: {resp.text[:100]}")
            return {"status": "success", "message": f"{dba_email}로 DBA 알림이 발송되었습니다."}
        except Exception as e:
            st.logger.error("[guardian-notify] resend error: %s", e)
            return error_response(str(e))
    else:
        st.logger.info("[guardian-notify] (RESEND_API_KEY 미설정 -- 시뮬레이션) -> %s", dba_email)
        return {"status": "success", "message": f"{dba_email}로 DBA 알림이 발송되었습니다. (RESEND_API_KEY 미설정 -- 시뮬레이션)"}
