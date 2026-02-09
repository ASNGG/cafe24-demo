// components/Sidebar.js
// CAFE24 AI 운영 플랫폼 사이드바
import { useState } from 'react';
import {
  LogOut,
  ChevronDown,
  ShoppingBag,
  Users,
  BarChart3,
  Search,
  MessageSquare,
  Sparkles,
  Building2,
  Package,
  X
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

function SidebarContent({
  auth,
  exampleQuestions,
  onExampleQuestion,
  onLogout,
  onClose,
  isMobile,
}) {
  const [openCats, setOpenCats] = useState({});

  // 카테고리별 스타일 (카페24 블루 테마)
  const CAT_STYLES = [
    { card: 'from-blue-50/70 to-white/70 border-blue-200/70 border-l-blue-400', icon: Package },
    { card: 'from-indigo-50/70 to-white/70 border-indigo-200/70 border-l-indigo-400', icon: Sparkles },
    { card: 'from-sky-50/70 to-white/70 border-sky-200/70 border-l-sky-400', icon: Search },
    { card: 'from-cyan-50/70 to-white/70 border-cyan-200/70 border-l-cyan-400', icon: Users },
    { card: 'from-violet-50/70 to-white/70 border-violet-200/70 border-l-violet-400', icon: BarChart3 },
  ];

  function clickExample(q) {
    onExampleQuestion(q);
    if (isMobile) onClose?.();
  }

  function toggleCat(cat) {
    setOpenCats((prev) => ({ ...prev, [cat]: !prev?.[cat] }));
  }

  const accordionVariants = {
    open: {
      height: 'auto',
      opacity: 1,
      transition: { duration: 0.24, ease: 'easeOut', when: 'beforeChildren', staggerChildren: 0.03 },
    },
    closed: {
      height: 0,
      opacity: 0,
      transition: { duration: 0.18, ease: 'easeIn', when: 'afterChildren' },
    },
  };

  const itemVariants = {
    open: { opacity: 1, y: 0, transition: { duration: 0.18, ease: 'easeOut' } },
    closed: { opacity: 0, y: -6, transition: { duration: 0.12, ease: 'easeIn' } },
  };

  const examples = exampleQuestions || {};

  return (
    <div className={isMobile ? 'h-full overflow-auto px-4 py-5' : 'px-4 py-5 pb-8'}>
      {/* 모바일 닫기 버튼 */}
      {isMobile && (
        <div className="flex justify-end mb-2">
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-cookie-orange/10 transition-colors"
          >
            <X className="w-5 h-5 text-cookie-brown" />
          </button>
        </div>
      )}

      {/* 로고 영역 */}
      <div className="pb-4 mb-4 border-b border-cookie-orange/20">
        <div className="flex items-start justify-between gap-2">
          <div className="inline-flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-white border border-cookie-orange/20 shadow-sm flex items-center justify-center overflow-hidden">
              <img src="https://img.echosting.cafe24.com/imgcafe24com/images/common/cafe24.svg" alt="CAFE24" className="w-8 h-8 object-contain" />
            </div>
            <div>
              <h2 className="text-base font-black text-cookie-brown leading-tight">CAFE24 Ops AI</h2>
              <p className="text-xs font-semibold text-cookie-orange">E-Commerce Platform</p>
            </div>
          </div>
        </div>
      </div>

      {/* 기능 소개 배지 */}
      <div className="mb-4 flex flex-wrap gap-2">
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-cookie-yellow/30 text-cookie-brown text-xs font-medium">
          <ShoppingBag className="w-3 h-3" /> 쇼핑몰
        </span>
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-cookie-orange/20 text-cookie-orange text-xs font-medium">
          <Search className="w-3 h-3" /> 분석
        </span>
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-blue-100 text-blue-700 text-xs font-medium">
          <BarChart3 className="w-3 h-3" /> CS
        </span>
      </div>

      {/* 사용자 정보 */}
      {auth?.username && (
        <div className="mb-4 p-3 rounded-xl bg-gradient-to-r from-cookie-yellow/20 to-cookie-orange/10 border border-cookie-orange/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cookie-yellow to-cookie-orange flex items-center justify-center">
                <Users className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="text-sm font-bold text-cookie-brown">{auth.user_name || auth.username}</p>
                <p className="text-xs text-cookie-orange">{auth.user_role || '사용자'}</p>
              </div>
            </div>
            <button
              onClick={onLogout}
              className="p-2 rounded-lg hover:bg-cookie-orange/10 transition-colors"
              title="로그아웃"
            >
              <LogOut className="w-4 h-4 text-cookie-brown/60" />
            </button>
          </div>
        </div>
      )}

      {/* 예시 질문 섹션 */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-4 h-4 text-cookie-orange" />
          <span className="text-sm font-bold text-cookie-brown">이렇게 물어보세요</span>
        </div>

        {Object.entries(examples).map(([cat, questions], catIdx) => {
          const style = CAT_STYLES[catIdx % CAT_STYLES.length];
          const isOpen = openCats[cat];

          return (
            <div key={cat} className="rounded-xl overflow-hidden">
              <button
                onClick={() => toggleCat(cat)}
                className={`w-full px-3 py-2.5 flex items-center justify-between bg-gradient-to-r ${style.card} border border-l-4 rounded-xl transition-all hover:shadow-sm`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-cookie-brown">{cat}</span>
                  <span className="text-xs text-cookie-brown/50">({questions.length})</span>
                </div>
                <ChevronDown
                  className={`w-4 h-4 text-cookie-brown/50 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
                />
              </button>

              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    initial="closed"
                    animate="open"
                    exit="closed"
                    variants={accordionVariants}
                    className="overflow-hidden"
                  >
                    <div className="pt-2 space-y-1.5">
                      {questions.map((q, idx) => (
                        <motion.button
                          key={idx}
                          variants={itemVariants}
                          onClick={() => clickExample(q)}
                          className="w-full text-left px-3 py-2 rounded-lg text-sm text-cookie-brown/80 hover:bg-cookie-yellow/20 hover:text-cookie-brown transition-colors border border-transparent hover:border-cookie-orange/20"
                        >
                          {q}
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {/* 하단 정보 */}
      <div className="mt-6 pt-4 border-t border-cookie-orange/20">
        <div className="text-center">
          <p className="text-xs text-cookie-brown/50 mb-2">Powered by</p>
          <div className="flex items-center justify-center gap-2">
            <Building2 className="w-4 h-4 text-cookie-orange" />
            <span className="text-sm font-bold bg-gradient-to-r from-cookie-orange to-cookie-yellow bg-clip-text text-transparent">
              CAFE24
            </span>
          </div>
          <p className="text-xs text-cookie-brown/40 mt-1">AI 기반 내부 시스템</p>
        </div>
      </div>
    </div>
  );
}

export default function Sidebar({
  auth,
  exampleQuestions,
  onExampleQuestion,
  onLogout,
  open,
  onClose,
  showWelcomePopup,
  onCloseWelcomePopup,
}) {
  return (
    <>
      {/* 로그인 환영 팝업 */}
      <AnimatePresence>
        {showWelcomePopup && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="hidden xl:block absolute top-24 left-0 z-50 px-3 w-full"
          >
            <div className="bg-white rounded-2xl shadow-2xl p-4 border-2 border-cookie-orange/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cookie-yellow to-cookie-orange flex items-center justify-center">
                    <MessageSquare className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-sm font-black text-cookie-brown">환영합니다!</h3>
                    <p className="text-xs text-cookie-orange">{auth?.user_name || auth?.username}님</p>
                  </div>
                </div>
                <button
                  onClick={onCloseWelcomePopup}
                  className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              </div>

              <div className="space-y-2 mb-3">
                <p className="text-xs text-cookie-brown/80">
                  이런 질문을 해보세요:
                </p>
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2 text-xs text-cookie-brown/70">
                    <span className="w-5 h-5 rounded-full bg-cookie-yellow/30 flex items-center justify-center text-[10px] font-bold">1</span>
                    <span>"S0001 쇼핑몰 정보 알려줘"</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-cookie-brown/70">
                    <span className="w-5 h-5 rounded-full bg-cookie-orange/30 flex items-center justify-center text-[10px] font-bold">2</span>
                    <span>"SEL0001 셀러 이탈 확률은?"</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-cookie-brown/70">
                    <span className="w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center text-[10px] font-bold">3</span>
                    <span>"오늘 GMV 현황 보여줘"</span>
                  </div>
                </div>
              </div>

              <button
                onClick={onCloseWelcomePopup}
                className="w-full py-2 rounded-xl bg-gradient-to-r from-cookie-orange to-cookie-yellow text-white text-sm font-bold shadow-md hover:shadow-lg transition-all"
              >
                시작하기
              </button>
            </div>
            <div className="absolute -bottom-2 left-8 w-4 h-4 bg-white border-r-2 border-b-2 border-cookie-orange/20 rotate-45" />
          </motion.div>
        )}
      </AnimatePresence>

      {/* 데스크탑 사이드바 */}
      <aside className="hidden xl:block sticky top-20 h-fit rounded-[32px] border-2 border-cookie-orange/10 bg-white/80 backdrop-blur-sm shadow-lg overflow-hidden">
        <SidebarContent
          auth={auth}
          exampleQuestions={exampleQuestions}
          onExampleQuestion={onExampleQuestion}
          onLogout={onLogout}
          isMobile={false}
        />
      </aside>

      {/* 모바일 사이드바 */}
      <AnimatePresence>
        {open && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="fixed inset-0 bg-black/30 z-40 xl:hidden"
            />
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 bottom-0 w-80 bg-gradient-to-b from-cookie-yellow/10 via-white to-cookie-orange/10 backdrop-blur-md z-50 xl:hidden shadow-2xl overflow-auto"
            >
              <SidebarContent
                auth={auth}
                exampleQuestions={exampleQuestions}
                onExampleQuestion={onExampleQuestion}
                onLogout={onLogout}
                onClose={onClose}
                isMobile={true}
              />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
