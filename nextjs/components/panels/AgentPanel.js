// components/panels/AgentPanel.js
// CAFE24 AI 운영 플랫폼 - 에이전트 패널 (Supervisor 모드 통합)

import MultiAgentPanel from './MultiAgentPanel';

export default function AgentPanel({
  auth, selectedShop, addLog, settings, apiCall,
}) {
  return (
    <MultiAgentPanel
      auth={auth}
      selectedShop={selectedShop}
      addLog={addLog}
      settings={settings}
      apiCall={apiCall}
    />
  );
}
