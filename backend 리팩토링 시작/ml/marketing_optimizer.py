"""
ml/marketing_optimizer.py - 마케팅 예산 최적화 (ML + P-PSO)
==========================================================
CAFE24 AI 운영 플랫폼

셀러의 마케팅 채널별 예산 배분을 최적화하여 최대 ROI 달성

핵심 기능:
1. 채널별 효율 계산: 각 마케팅 채널의 예상 ROAS 및 매출 증가 효과
2. P-PSO 최적화: 총 예산 제약 하에서 최적 예산 배분 탐색
3. 개인화: 셀러마다 다른 성과 데이터 → 다른 추천 결과
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

# 프로젝트 루트
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # 주피터 노트북 실행 시
    if 'BACKEND_DIR' in dir():
        PROJECT_ROOT = BACKEND_DIR
    else:
        _cwd = Path(".").resolve()
        if _cwd.name == "ml":
            PROJECT_ROOT = _cwd.parent
        else:
            PROJECT_ROOT = _cwd

# H22: CSV 데이터 캐시 (매 호출 시 중복 I/O 방지)
_CSV_CACHE: Dict[str, pd.DataFrame] = {}


def _load_csv_cached(path: Path) -> Optional[pd.DataFrame]:
    """CSV 로드 결과를 캐싱 (H22: 매 호출 시 중복 I/O 방지)"""
    key = str(path)
    if key in _CSV_CACHE:
        return _CSV_CACHE[key]
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        _CSV_CACHE[key] = df
        return df
    except Exception as e:
        logger.warning(f"CSV load failed: {path} - {e}")
        return None

# ========================================
# 마케팅 채널 정의 및 기본 파라미터
# ========================================

# 마케팅 채널 목록
MARKETING_CHANNELS = [
    'search_ads',         # 검색 광고 (네이버, 구글)
    'display_ads',        # 디스플레이 광고 (배너, GDN)
    'social_media',       # 소셜미디어 (인스타, 페이스북)
    'email_marketing',    # 이메일 마케팅
    'influencer',         # 인플루언서 마케팅
    'content_marketing',  # 콘텐츠 마케팅 (블로그, SEO)
]

# 채널별 기본 파라미터 (셀러 데이터가 없을 때 기본값)
DEFAULT_CHANNEL_PARAMS = {
    'search_ads': {
        'name': '검색 광고',
        'min_budget': 100_000,        # 최소 예산 (원)
        'max_budget_ratio': 0.40,     # 전체 예산 대비 최대 비율
        'expected_roas': 3.5,         # 기대 ROAS (1원 투자 → 3.5원 매출)
        'expected_revenue_uplift': 0.08,  # 기대 매출 증가율 (8%)
        'saturation_point': 0.30,     # 수확체감 시작 비율
    },
    'display_ads': {
        'name': '디스플레이 광고',
        'min_budget': 50_000,
        'max_budget_ratio': 0.30,
        'expected_roas': 2.0,
        'expected_revenue_uplift': 0.05,
        'saturation_point': 0.25,
    },
    'social_media': {
        'name': '소셜미디어 광고',
        'min_budget': 50_000,
        'max_budget_ratio': 0.35,
        'expected_roas': 4.0,
        'expected_revenue_uplift': 0.10,
        'saturation_point': 0.28,
    },
    'email_marketing': {
        'name': '이메일 마케팅',
        'min_budget': 20_000,
        'max_budget_ratio': 0.15,
        'expected_roas': 8.0,
        'expected_revenue_uplift': 0.04,
        'saturation_point': 0.10,
    },
    'influencer': {
        'name': '인플루언서 마케팅',
        'min_budget': 200_000,
        'max_budget_ratio': 0.30,
        'expected_roas': 2.5,
        'expected_revenue_uplift': 0.07,
        'saturation_point': 0.25,
    },
    'content_marketing': {
        'name': '콘텐츠 마케팅',
        'min_budget': 30_000,
        'max_budget_ratio': 0.20,
        'expected_roas': 5.0,
        'expected_revenue_uplift': 0.06,
        'saturation_point': 0.15,
    },
}


class MarketingOptimizer:
    """마케팅 예산 최적화기"""

    def __init__(self, seller_id: str, total_budget: float, goal: str = 'balanced'):
        """
        Args:
            seller_id: 셀러 ID
            total_budget: 총 마케팅 예산 (원)
            goal: 'maximize_roas' | 'maximize_revenue' | 'balanced'
        """
        self.seller_id = seller_id
        self.total_budget = total_budget
        self.goal = goal

        self.seller_data = None
        self.seller_analytics = None
        self.channel_params = None
        self.revenue_predictor = None

        self._load_data()
        self._load_revenue_model()
        self._initialize_channel_params()

    def _load_data(self):
        """셀러 데이터 로딩 (H22: 캐시 활용으로 중복 I/O 제거)"""
        try:
            # 셀러 기본 데이터
            all_sellers = _load_csv_cached(PROJECT_ROOT / "sellers.csv")
            if all_sellers is not None:
                seller_row = all_sellers[all_sellers['seller_id'] == self.seller_id]
                if len(seller_row) > 0:
                    self.seller_data = seller_row.iloc[0].to_dict()
                    logger.info(f"Loaded seller data for {self.seller_id}")
                else:
                    logger.warning(f"Seller {self.seller_id} not found in sellers.csv")
                    self.seller_data = {}
            else:
                logger.warning("sellers.csv not found")
                self.seller_data = {}

            # 셀러 분석 데이터
            all_analytics = _load_csv_cached(PROJECT_ROOT / "seller_analytics.csv")
            if all_analytics is not None:
                analytics_row = all_analytics[all_analytics['seller_id'] == self.seller_id]
                if len(analytics_row) > 0:
                    self.seller_analytics = analytics_row.iloc[0].to_dict()
                    logger.info(f"Loaded analytics for {self.seller_id}")
                else:
                    logger.warning(f"Seller {self.seller_id} not found in seller_analytics.csv")
                    self.seller_analytics = {}
            else:
                logger.warning("seller_analytics.csv not found")
                self.seller_analytics = {}

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _load_revenue_model(self):
        """매출 예측 모델 로딩"""
        try:
            from ml.revenue_model import get_predictor
            self.revenue_predictor = get_predictor()
            if self.revenue_predictor and self.revenue_predictor.is_fitted:
                logger.info("Revenue predictor loaded successfully")
            else:
                logger.warning("Revenue predictor not fitted")
                self.revenue_predictor = None
        except Exception as e:
            logger.warning(f"Failed to load revenue model: {e}")
            self.revenue_predictor = None

    def _initialize_channel_params(self):
        """
        셀러 분석 데이터 기반으로 채널 파라미터 개인화

        셀러의 과거 마케팅 성과가 있으면 해당 데이터로 ROAS/효과 업데이트
        없으면 기본값 사용
        """
        self.channel_params = {}

        for channel in MARKETING_CHANNELS:
            # 기본 파라미터 복사
            params = DEFAULT_CHANNEL_PARAMS[channel].copy()

            # 셀러 분석 데이터에서 채널별 성과 반영
            if self.seller_analytics:
                # 예: seller_analytics.csv에 search_ads_roas, social_media_roas 등의 컬럼이 있을 때
                roas_key = f'{channel}_roas'
                uplift_key = f'{channel}_uplift'

                if roas_key in self.seller_analytics and not pd.isna(self.seller_analytics.get(roas_key)):
                    params['expected_roas'] = float(self.seller_analytics[roas_key])
                if uplift_key in self.seller_analytics and not pd.isna(self.seller_analytics.get(uplift_key)):
                    params['expected_revenue_uplift'] = float(self.seller_analytics[uplift_key])

            self.channel_params[channel] = params

    def calculate_channel_efficiency(
        self,
        channel: str,
        budget_allocated: float
    ) -> Dict[str, Any]:
        """
        특정 채널에 예산 배분 시 효율 계산

        수확체감(diminishing returns) 모델 적용:
        - 예산이 saturation_point 이하일 때: 선형 수익
        - 예산이 saturation_point 초과일 때: 로그 수익 (체감)

        Args:
            channel: 마케팅 채널 이름
            budget_allocated: 배분 예산 (원)

        Returns:
            {
                'channel': str,
                'budget': float,
                'expected_revenue': float,
                'expected_roas': float,
                'expected_revenue_uplift': float,
                'efficiency_score': float,
            }
        """
        params = self.channel_params.get(channel, DEFAULT_CHANNEL_PARAMS.get(channel))
        if params is None:
            return {'error': f'Unknown channel: {channel}'}

        if budget_allocated <= 0:
            return {
                'channel': channel,
                'channel_name': params['name'],
                'budget': 0,
                'expected_revenue': 0,
                'expected_roas': 0,
                'expected_revenue_uplift': 0,
                'efficiency_score': 0,
            }

        base_roas = params['expected_roas']
        saturation_budget = self.total_budget * params['saturation_point']

        # 수확체감 모델: saturation_point 넘으면 효율 감소
        if budget_allocated <= saturation_budget:
            # 선형 구간: ROAS 유지
            effective_roas = base_roas
        else:
            # 체감 구간: 로그 스케일로 감소
            over_ratio = budget_allocated / saturation_budget
            effective_roas = base_roas * (1.0 / (1.0 + 0.3 * np.log(over_ratio)))

        expected_revenue = budget_allocated * effective_roas
        expected_uplift = params['expected_revenue_uplift'] * (budget_allocated / (self.total_budget * 0.2))
        expected_uplift = min(expected_uplift, params['expected_revenue_uplift'] * 2)  # 상한

        # 효율 점수: ROAS * 매출증가율 (정규화)
        efficiency_score = effective_roas * expected_uplift * 100

        return {
            'channel': channel,
            'channel_name': params['name'],
            'budget': round(budget_allocated),
            'expected_revenue': round(expected_revenue),
            'expected_roas': round(effective_roas, 2),
            'expected_revenue_uplift': round(expected_uplift, 4),
            'efficiency_score': round(efficiency_score, 4),
        }

    def optimize(
        self,
        max_iterations: int = 200,
        population_size: int = 50,
    ) -> Dict[str, Any]:
        """
        P-PSO로 최적 마케팅 예산 배분 탐색

        Args:
            max_iterations: P-PSO 반복 횟수 (기본 200)
            population_size: P-PSO 개체 수 (기본 50)

        Returns:
            {
                'seller_id': str,
                'total_budget': float,
                'goal': str,
                'allocation': [...],  # 채널별 예산 배분
                'total_expected_revenue': float,
                'overall_roas': float,
                'optimization_method': str,
            }
        """
        if self.total_budget <= 0:
            return {'error': '총 마케팅 예산이 0 이하입니다'}

        # 최소 예산 합계 체크
        total_min_budget = sum(
            self.channel_params[ch]['min_budget'] for ch in MARKETING_CHANNELS
        )
        if self.total_budget < total_min_budget:
            # 예산이 너무 적으면 상위 채널만 추천
            return self._recommend_limited_budget()

        # P-PSO 최적화 시도
        try:
            allocation = self._run_pso_optimization(max_iterations, population_size)
        except Exception as e:
            logger.warning(f"PSO optimization failed: {e}, using heuristic fallback")
            allocation = self._heuristic_allocation()

        # 결과 조합
        results = []
        for channel, budget in allocation.items():
            if budget > 0:
                channel_result = self.calculate_channel_efficiency(channel, budget)
                results.append(channel_result)

        # 매출 예측 모델이 있으면 예측 매출도 포함
        predicted_next_revenue = None
        if self.revenue_predictor and self.seller_analytics:
            try:
                features = {
                    'total_revenue': self.seller_analytics.get('total_revenue', 0),
                    'total_orders': self.seller_analytics.get('total_orders', 0),
                    'unique_customers': self.seller_analytics.get('unique_customers', 0),
                    'avg_order_value': self.seller_analytics.get('avg_order_value', 0),
                    'revenue_growth': self.seller_analytics.get('revenue_growth', 0),
                    'conversion_rate': self.seller_analytics.get('conversion_rate', 0),
                    'review_score': self.seller_analytics.get('review_score', 0),
                }
                predicted_next_revenue = self.revenue_predictor.predict(features)
            except Exception as e:
                logger.warning(f"Revenue prediction failed: {e}")

        total_expected_revenue = sum(r['expected_revenue'] for r in results)
        total_allocated = sum(r['budget'] for r in results)
        overall_roas = total_expected_revenue / total_allocated if total_allocated > 0 else 0

        # 예산 사용률
        budget_usage = {
            'total_budget': self.total_budget,
            'total_allocated': total_allocated,
            'usage_pct': round(total_allocated / self.total_budget * 100, 1) if self.total_budget > 0 else 0,
            'remaining': round(self.total_budget - total_allocated),
        }

        output = {
            'seller_id': self.seller_id,
            'total_budget': self.total_budget,
            'goal': self.goal,
            'allocation': sorted(results, key=lambda x: x['budget'], reverse=True),
            'total_expected_revenue': round(total_expected_revenue),
            'overall_roas': round(overall_roas, 2),
            'budget_usage': budget_usage,
            'optimization_method': 'P-PSO (Phasor Particle Swarm Optimization)',
        }

        if predicted_next_revenue is not None:
            output['predicted_next_month_revenue'] = round(predicted_next_revenue)

        return output

    def _recommend_limited_budget(self) -> Dict[str, Any]:
        """
        예산이 적을 때 상위 효율 채널만 추천

        ROAS가 높은 순으로 예산 배분
        """
        # ROAS 순으로 정렬
        sorted_channels = sorted(
            MARKETING_CHANNELS,
            key=lambda ch: self.channel_params[ch]['expected_roas'],
            reverse=True
        )

        allocation = {}
        remaining_budget = self.total_budget

        for channel in sorted_channels:
            params = self.channel_params[channel]
            min_budget = params['min_budget']

            if remaining_budget >= min_budget:
                # 최소 예산 배분
                allocated = min(remaining_budget, min_budget * 2)  # 최소의 2배까지
                allocation[channel] = allocated
                remaining_budget -= allocated

            if remaining_budget <= 0:
                break

        results = []
        for channel, budget in allocation.items():
            channel_result = self.calculate_channel_efficiency(channel, budget)
            results.append(channel_result)

        total_expected_revenue = sum(r['expected_revenue'] for r in results)
        total_allocated = sum(r['budget'] for r in results)
        overall_roas = total_expected_revenue / total_allocated if total_allocated > 0 else 0

        return {
            'seller_id': self.seller_id,
            'total_budget': self.total_budget,
            'goal': self.goal,
            'allocation': sorted(results, key=lambda x: x['budget'], reverse=True),
            'total_expected_revenue': round(total_expected_revenue),
            'overall_roas': round(overall_roas, 2),
            'budget_usage': {
                'total_budget': self.total_budget,
                'total_allocated': total_allocated,
                'usage_pct': round(total_allocated / self.total_budget * 100, 1),
                'remaining': round(self.total_budget - total_allocated),
            },
            'optimization_method': 'Heuristic (limited budget)',
            'note': '예산이 제한적이어서 고효율 채널 위주로 배분하였습니다.',
        }

    def _heuristic_allocation(self) -> Dict[str, float]:
        """
        P-PSO 실패 시 휴리스틱 예산 배분 (fallback)

        전략: 각 채널의 expected_roas 비율로 예산 배분
        """
        total_roas = sum(
            self.channel_params[ch]['expected_roas'] for ch in MARKETING_CHANNELS
        )

        allocation = {}
        remaining = self.total_budget

        for channel in MARKETING_CHANNELS:
            params = self.channel_params[channel]
            ratio = params['expected_roas'] / total_roas
            max_budget = self.total_budget * params['max_budget_ratio']
            min_budget = params['min_budget']

            budget = max(min_budget, min(self.total_budget * ratio, max_budget))
            budget = min(budget, remaining)
            allocation[channel] = budget
            remaining -= budget

        # 남은 예산을 ROAS 높은 순으로 추가 배분
        if remaining > 0:
            sorted_channels = sorted(
                MARKETING_CHANNELS,
                key=lambda ch: self.channel_params[ch]['expected_roas'],
                reverse=True
            )
            for channel in sorted_channels:
                params = self.channel_params[channel]
                max_budget = self.total_budget * params['max_budget_ratio']
                can_add = max_budget - allocation.get(channel, 0)
                add_amount = min(remaining, can_add)
                if add_amount > 0:
                    allocation[channel] = allocation.get(channel, 0) + add_amount
                    remaining -= add_amount
                if remaining <= 0:
                    break

        return allocation

    def _run_pso_optimization(
        self,
        max_iterations: int,
        population_size: int,
    ) -> Dict[str, float]:
        """
        P-PSO 최적화 실행 - 연속 변수로 예산 비율 탐색

        각 차원: 채널별 예산 비율 (0.0 ~ max_budget_ratio)
        제약: 합계 <= 1.0 (100%)
        """
        from mealpy.swarm_based.PSO import P_PSO
        from mealpy.utils.space import FloatVar
        logger.info(f"Using P_PSO with {max_iterations} iterations, pop_size={population_size}")

        n_channels = len(MARKETING_CHANNELS)

        if n_channels == 0:
            raise ValueError("최적화할 채널이 없습니다")

        # 채널별 예산 비율 범위 설정
        lower_bounds = []
        upper_bounds = []
        for channel in MARKETING_CHANNELS:
            params = self.channel_params[channel]
            min_ratio = params['min_budget'] / self.total_budget if self.total_budget > 0 else 0
            max_ratio = params['max_budget_ratio']
            lower_bounds.append(min_ratio)
            upper_bounds.append(max_ratio)

        # M31: numpy vectorized fitness - 채널 파라미터 사전 추출
        _min_budgets = np.array([self.channel_params[ch]['min_budget'] for ch in MARKETING_CHANNELS])
        _base_roas = np.array([self.channel_params[ch]['expected_roas'] for ch in MARKETING_CHANNELS])
        _sat_points = np.array([self.channel_params[ch]['saturation_point'] for ch in MARKETING_CHANNELS])
        _uplifts = np.array([self.channel_params[ch]['expected_revenue_uplift'] for ch in MARKETING_CHANNELS])

        def fitness_function(solution):
            """M31: numpy vectorized fitness function"""
            ratios = np.array(solution)

            total_ratio = np.sum(ratios)
            if total_ratio > 1.05:
                return -1000 * (total_ratio - 1.0)

            if total_ratio > 1.0:
                ratios = ratios / total_ratio

            budgets = self.total_budget * ratios
            active_mask = budgets >= _min_budgets

            # Vectorized ROAS 계산 (수확체감 모델)
            sat_budgets = self.total_budget * _sat_points
            over_ratios = np.where(sat_budgets > 0, budgets / sat_budgets, 1.0)
            effective_roas = np.where(
                budgets <= sat_budgets,
                _base_roas,
                _base_roas * (1.0 / (1.0 + 0.3 * np.log(np.maximum(over_ratios, 1.0))))
            )

            expected_rev = budgets * effective_roas * active_mask
            expected_uplift = np.minimum(
                _uplifts * (budgets / (self.total_budget * 0.2)),
                _uplifts * 2
            ) * active_mask

            total_revenue = np.sum(expected_rev)
            total_cost = np.sum(budgets * active_mask)

            if self.goal == 'maximize_roas':
                total_score = np.sum(effective_roas * budgets * active_mask)
            elif self.goal == 'maximize_revenue':
                total_score = total_revenue
            else:  # balanced
                total_score = np.sum(
                    (effective_roas * 0.4 + expected_uplift * 100 * 0.3) * budgets * active_mask
                    + expected_rev * 0.3
                )

            utilization = total_cost / self.total_budget if self.total_budget > 0 else 0
            if utilization < 0.7:
                total_score *= utilization

            return float(total_score)

        # mealpy 3.x 방식: FloatVar 사용
        bounds = FloatVar(lb=lower_bounds, ub=upper_bounds, name="budget_ratios")

        problem = {
            "obj_func": fitness_function,
            "bounds": bounds,
            "minmax": "max",
        }

        # P_PSO 실행
        model = P_PSO(epoch=max_iterations, pop_size=population_size)
        best_agent = model.solve(problem)
        best_solution = np.array(best_agent.solution)

        # 정규화
        total_ratio = np.sum(best_solution)
        if total_ratio > 1.0:
            best_solution = best_solution / total_ratio

        # 결과 변환: 비율 → 실제 예산
        allocation = {}
        for i, channel in enumerate(MARKETING_CHANNELS):
            params = self.channel_params[channel]
            budget = self.total_budget * best_solution[i]

            # 최소 예산 미달 시 0 처리
            if budget < params['min_budget']:
                budget = 0

            allocation[channel] = round(budget)

        # 최종 검증: 총 배분액이 예산 초과 시 비례 축소
        total_allocated = sum(allocation.values())
        if total_allocated > self.total_budget:
            scale_factor = self.total_budget / total_allocated
            allocation = {ch: round(b * scale_factor) for ch, b in allocation.items()}

        total_revenue = sum(
            self.calculate_channel_efficiency(ch, b).get('expected_revenue', 0)
            for ch, b in allocation.items() if b > 0
        )
        logger.info(
            f"P_PSO 최적화 완료: {sum(1 for b in allocation.values() if b > 0)}개 채널, "
            f"총 배분 {sum(allocation.values()):,}원, 예상 매출 {total_revenue:,}원"
        )

        return allocation


# ========================================
# 유틸리티 함수
# ========================================

def get_marketing_recommendations(
    seller_id: str,
    total_budget: float,
    goal: str = 'balanced'
) -> Dict:
    """셀러에게 마케팅 예산 배분 추천 제공"""
    try:
        optimizer = MarketingOptimizer(seller_id, total_budget, goal)
        return optimizer.optimize()
    except Exception as e:
        logger.error(f"Optimization failed for seller {seller_id}: {e}")
        return {'error': str(e)}


def compare_sellers(
    seller_ids: List[str],
    total_budget: float,
    goal: str = 'balanced'
) -> Dict:
    """여러 셀러의 추천 결과 비교 (개인화 검증용)"""
    results = {}
    for seller_id in seller_ids:
        results[seller_id] = get_marketing_recommendations(seller_id, total_budget, goal)
    return results


# ========================================
# 테스트
# ========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("마케팅 예산 최적화 테스트")
    print("CAFE24 AI 운영 플랫폼")
    print("=" * 60)

    # 테스트 셀러
    test_seller = "S00001"
    test_budget = 5_000_000  # 500만원

    print(f"\n셀러 {test_seller}에 대한 마케팅 예산 최적화...")
    print(f"총 예산: {test_budget:,}원")

    # 목표별 테스트
    for goal in ['maximize_roas', 'maximize_revenue', 'balanced']:
        print(f"\n{'=' * 40}")
        print(f"목표: {goal}")
        print(f"{'=' * 40}")

        result = get_marketing_recommendations(test_seller, test_budget, goal)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n[추천 결과]")
            print(f"최적화 방법: {result['optimization_method']}")
            print(f"전체 예상 ROAS: {result['overall_roas']}")
            print(f"전체 예상 매출: {result['total_expected_revenue']:,}원")

            if 'predicted_next_month_revenue' in result:
                print(f"다음 달 예측 매출: {result['predicted_next_month_revenue']:,}원")

            print(f"\n[채널별 예산 배분 ({len(result['allocation'])}개 채널)]")
            for i, alloc in enumerate(result['allocation'], 1):
                print(f"  {i}. {alloc['channel_name']} ({alloc['channel']})")
                print(f"     - 예산: {alloc['budget']:,}원")
                print(f"     - 예상 ROAS: {alloc['expected_roas']}")
                print(f"     - 예상 매출: {alloc['expected_revenue']:,}원")
                print(f"     - 매출 증가 효과: {alloc['expected_revenue_uplift']:.2%}")

            usage = result['budget_usage']
            print(f"\n[예산 사용률]")
            print(f"  총 배분: {usage['total_allocated']:,}원 / {usage['total_budget']:,}원 ({usage['usage_pct']}%)")
            print(f"  잔여: {usage['remaining']:,}원")

    # 개인화 검증: 다른 셀러와 비교
    print("\n" + "=" * 60)
    print("개인화 검증 (셀러별 추천 비교)")
    print("=" * 60)

    comparison = compare_sellers(["S00001", "S00002", "S00003"], test_budget)
    for seller_id, res in comparison.items():
        if 'error' in res:
            print(f"\n{seller_id}: Error - {res['error']}")
        else:
            top_alloc = res['allocation'][0] if res['allocation'] else None
            if top_alloc:
                print(f"\n{seller_id} 최대 배분 채널: {top_alloc['channel_name']} "
                      f"({top_alloc['budget']:,}원, ROAS {top_alloc['expected_roas']})")
            else:
                print(f"\n{seller_id}: 추천 없음")
