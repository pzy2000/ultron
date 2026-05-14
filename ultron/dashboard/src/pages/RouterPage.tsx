import { useEffect, useState } from 'react';
import { api, apiPost } from '../api/client';

interface RouterSettings {
  enabled: boolean;
  model: string;
  base_url: string;
  has_api_key: boolean;
}

const DEFAULT_MESSAGES = JSON.stringify(
  [{ role: 'user', content: 'Reply with a concise hello from Ultron router.' }],
  null,
  2,
);

export default function RouterPage() {
  const [settings, setSettings] = useState<RouterSettings | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [model, setModel] = useState('qwen3.6-plus');
  const [baseUrl, setBaseUrl] = useState('https://dashscope.aliyuncs.com/compatible-mode/v1');
  const [apiKey, setApiKey] = useState('');
  const [messages, setMessages] = useState(DEFAULT_MESSAGES);
  const [temperature, setTemperature] = useState('0.2');
  const [maxTokens, setMaxTokens] = useState('256');
  const [status, setStatus] = useState('');
  const [testResult, setTestResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api('/router/settings')
      .then(j => {
        const data = j.data || j;
        setSettings(data);
        setEnabled(!!data.enabled);
        setModel(data.model || 'qwen3.6-plus');
        setBaseUrl(data.base_url || 'https://dashscope.aliyuncs.com/compatible-mode/v1');
      })
      .catch(e => setStatus(String(e?.message || e)));
  }, []);

  const saveSettings = async () => {
    setBusy(true);
    setStatus('');
    try {
      const body: any = { enabled, model, base_url: baseUrl };
      if (apiKey.trim()) body.api_key = apiKey.trim();
      const j = await apiPost('/router/settings', body);
      if (!j.success) throw new Error(j.detail || j.error || 'Save failed');
      setSettings(j.data);
      setApiKey('');
      setStatus('Saved');
    } catch (e: any) {
      setStatus(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const runTest = async () => {
    setBusy(true);
    setStatus('');
    setTestResult(null);
    try {
      const parsed = JSON.parse(messages);
      const started = performance.now();
      const j = await apiPost('/v1/chat/completions', {
        model,
        messages: parsed,
        temperature: Number(temperature),
        max_tokens: Number(maxTokens),
        stream: false,
      });
      const latency = Math.round(performance.now() - started);
      setTestResult({ latency, response: j });
      if (j.error) setStatus(j.error.message || 'Router test failed');
    } catch (e: any) {
      setStatus(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const output = testResult?.response?.choices?.[0]?.message?.content || '';
  const error = testResult?.response?.error?.message || '';

  return (
    <div className="p-6 space-y-6 max-w-[1100px] mx-auto">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold font-serif">Router</h1>
          <p className="text-sm text-muted mt-1">OpenAI-compatible routing backed by the Ultron server model.</p>
        </div>
        <span className={enabled ? 'tag tag-hot' : 'tag tag-cold'}>
          {enabled ? 'Enabled' : 'Disabled'}
        </span>
      </div>

      <section className="panel-surface p-5 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="space-y-1">
            <span className="kicker">Model</span>
            <input className="w-full" value={model} onChange={e => setModel(e.target.value)} />
          </label>
          <label className="space-y-1">
            <span className="kicker">API Base</span>
            <input className="w-full" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} />
          </label>
          <label className="space-y-1">
            <span className="kicker">API Key</span>
            <input
              className="w-full"
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={settings?.has_api_key ? 'Configured; leave blank to keep current key' : 'Optional server-side key'}
              autoComplete="off"
            />
          </label>
          <div className="flex items-end">
            <label className="inline-flex items-center gap-3 text-sm">
              <input
                type="checkbox"
                checked={enabled}
                onChange={e => setEnabled(e.target.checked)}
                className="h-4 w-4"
              />
              Router enabled
            </label>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button className="btn-primary" disabled={busy} onClick={saveSettings}>Save Settings</button>
          <span className="text-sm text-muted">
            API key status: {settings?.has_api_key ? 'configured' : 'not configured'}
          </span>
        </div>
      </section>

      <section className="panel-surface p-5 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_minmax(120px,160px)_minmax(120px,160px)] gap-4">
          <label className="min-w-0 space-y-1">
            <span className="kicker">Messages JSON</span>
            <textarea
              className="w-full min-h-[180px] font-mono"
              value={messages}
              onChange={e => setMessages(e.target.value)}
            />
          </label>
          <label className="min-w-0 space-y-1">
            <span className="kicker">Temperature</span>
            <input className="w-full" value={temperature} onChange={e => setTemperature(e.target.value)} />
          </label>
          <label className="min-w-0 space-y-1">
            <span className="kicker">Max Tokens</span>
            <input className="w-full" value={maxTokens} onChange={e => setMaxTokens(e.target.value)} />
          </label>
        </div>
        <button className="btn-outline" disabled={busy} onClick={runTest}>Test /v1/chat/completions</button>

        {(status || output || error) && (
          <div className="record-card bg-surface p-4 space-y-3">
            {status && <div className={error ? 'text-danger text-sm' : 'text-muted text-sm'}>{status}</div>}
            {testResult && (
              <div className="flex flex-wrap gap-2">
                <span className="tag tag-type">{testResult.latency} ms</span>
                <span className="tag tag-type">{testResult.response?.model || model}</span>
              </div>
            )}
            {output && (
              <pre className="p-3 bg-paper rounded-card-sm text-xs whitespace-pre-wrap break-words">{output}</pre>
            )}
            {error && (
              <pre className="p-3 bg-paper rounded-card-sm text-xs whitespace-pre-wrap break-words text-danger">{error}</pre>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
