"use client";
import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import LogoBaufest from "../Logo_Baufest_PNG.png";
import LogoBaufestWhite from "../logo_blanco.png";

 // Base URL for backend API, configurable via env
 const API_BASE =
   process.env.NEXT_PUBLIC_BACKEND_URL ||
   process.env.NEXT_PUBLIC_API_BASE ||
   process.env.BACKEND_URL ||
   "http://localhost:8000";

export default function Page() {
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [showSources, setShowSources] = useState(false); // por defecto desmarcado
  const [recording, setRecording] = useState(false);
  const [speakReply, setSpeakReply] = useState(true);
  
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  // STT por lotes (batch) únicamente
  const ttsAudioRef = useRef(null);
  const ttsAbortRef = useRef(null);
  const [ttsActive, setTtsActive] = useState(false); // activo durante descarga o reproducción
  const ttsQueueRef = useRef([]); // URLs ya prefeteadas para los siguientes chunks
  const ttsCtrlsRef = useRef([]); // AbortControllers de cada fetch de chunk
  const [theme, setTheme] = useState('baufest');
  // Configuración de video OCR
  const [frameEverySecs, setFrameEverySecs] = useState(3);
  const [maxFrames, setMaxFrames] = useState(200); // 0 = sin límite
  const [sceneDetection, setSceneDetection] = useState(true);
  const [doOcrFrames, setDoOcrFrames] = useState(false);
  const [ocrEngine, setOcrEngine] = useState('azure'); // 'azure' | 'tesseract'
  // Modal de configuración
  const [showSettings, setShowSettings] = useState(false);
  // Modal Acerca de
  const [showAbout, setShowAbout] = useState(false);

  // Efecto para desplazarse al final del chat cuando hay nuevos mensajes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // Inicializar tema desde query/localStorage/env
  useEffect(() => {
    try {
      const url = new URL(window.location.href);
      const qsTheme = url.searchParams.get('theme');
      const lsTheme = localStorage.getItem('ui_theme');
      const envTheme = process.env.NEXT_PUBLIC_UI_STYLE;
      const initial = (qsTheme || lsTheme || envTheme || 'baufest').toLowerCase();
      const valid = initial === 'baufest' ? 'baufest' : 'classic';
      setTheme(valid);
      document.documentElement.setAttribute('data-theme', valid);
    } catch {}
  }, []);

  // Cargar configuración desde localStorage
  useEffect(() => {
    try {
      const lsShow = localStorage.getItem('cfg_show_sources');
      if (lsShow !== null) setShowSources(lsShow === 'true');
      const lsSpeak = localStorage.getItem('cfg_speak_reply');
      if (lsSpeak !== null) setSpeakReply(lsSpeak === 'true');
      const lsFes = localStorage.getItem('cfg_frame_secs');
      if (lsFes !== null) setFrameEverySecs(Math.max(1, parseInt(lsFes, 10) || 1));
      const lsMax = localStorage.getItem('cfg_max_frames');
      if (lsMax !== null) setMaxFrames(Math.max(0, parseInt(lsMax, 10) || 0));
      const lsScene = localStorage.getItem('cfg_scene_detection');
      if (lsScene !== null) setSceneDetection(lsScene === 'true');
      const lsDoOcr = localStorage.getItem('cfg_do_ocr_frames');
      if (lsDoOcr !== null) setDoOcrFrames(lsDoOcr === 'true');
      const lsOcr = localStorage.getItem('cfg_ocr_engine');
      if (lsOcr) setOcrEngine(lsOcr === 'tesseract' ? 'tesseract' : 'azure');
    } catch {}
  }, []);

  // Guardar configuración en localStorage
  useEffect(() => { try { localStorage.setItem('cfg_show_sources', String(showSources)); } catch {} }, [showSources]);
  useEffect(() => { try { localStorage.setItem('cfg_speak_reply', String(speakReply)); } catch {} }, [speakReply]);
  useEffect(() => { try { localStorage.setItem('cfg_frame_secs', String(frameEverySecs)); } catch {} }, [frameEverySecs]);
  useEffect(() => { try { localStorage.setItem('cfg_max_frames', String(maxFrames)); } catch {} }, [maxFrames]);
  useEffect(() => { try { localStorage.setItem('cfg_scene_detection', String(sceneDetection)); } catch {} }, [sceneDetection]);
  useEffect(() => { try { localStorage.setItem('cfg_do_ocr_frames', String(doOcrFrames)); } catch {} }, [doOcrFrames]);
  useEffect(() => { try { localStorage.setItem('cfg_ocr_engine', String(ocrEngine)); } catch {} }, [ocrEngine]);

  // Cargar configuración inicial desde backend (.env)
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/config`);
        if (!res.ok) return;
        const cfg = await res.json();
        if (typeof cfg.FRAME_EVERY_SECS === 'number') setFrameEverySecs(Math.max(1, cfg.FRAME_EVERY_SECS));
        if (typeof cfg.MAX_FRAMES === 'number') setMaxFrames(Math.max(0, cfg.MAX_FRAMES));
        if (typeof cfg.SCENE_DETECTION === 'boolean') setSceneDetection(!!cfg.SCENE_DETECTION);
        if (typeof cfg.DO_OCR_FRAMES === 'boolean') setDoOcrFrames(!!cfg.DO_OCR_FRAMES);
        if (typeof cfg.SHOW_SOURCES === 'boolean') setShowSources(!!cfg.SHOW_SOURCES);
        if (typeof cfg.SPEAK_REPLY === 'boolean') setSpeakReply(!!cfg.SPEAK_REPLY);
        if (cfg.OCR_ENGINE === 'tesseract' || cfg.OCR_ENGINE === 'azure') setOcrEngine(cfg.OCR_ENGINE);
      } catch {}
    })();
  }, []);

// --- Mermaid rendering helpers ---
function splitMermaidBlocks(text) {
  try {
    const parts = [];
    const regex = /```mermaid\s*([\s\S]*?)```/gi;
    let lastIndex = 0;
    let m;
    while ((m = regex.exec(text)) !== null) {
      const start = m.index;
      const end = regex.lastIndex;
      if (start > lastIndex) {
        parts.push({ type: 'text', value: text.slice(lastIndex, start) });
      }
      parts.push({ type: 'mermaid', value: (m[1] || '').trim() });
      lastIndex = end;
    }
    if (lastIndex < text.length) {
      parts.push({ type: 'text', value: text.slice(lastIndex) });
    }
    if (parts.length === 0) return [{ type: 'text', value: text }];
    return parts;
  } catch {
    return [{ type: 'text', value: text }];
  }
}

function AssistantContent({ content, msgIndex }) {
  try {
    const parts = splitMermaidBlocks(content || '');
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {parts.map((p, i) =>
          p.type === 'mermaid' ? (
            <MermaidDiagram key={`m-${msgIndex}-${i}`} code={p.value} idPrefix={`m-${msgIndex}-${i}`} />
          ) : (
            <div key={`t-${msgIndex}-${i}`} dangerouslySetInnerHTML={renderMarkdown(p.value)} />
          )
        )}
      </div>
    );
  } catch {
    return <div dangerouslySetInnerHTML={renderMarkdown(content || '')} />;
  }
}

function MermaidDiagram({ code, idPrefix }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: 'default' });
        const { svg } = await mermaid.render(`${idPrefix}`, code);
        if (cancelled) return;
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
          const svgEl = containerRef.current.querySelector('svg');
          if (svgEl) {
            // Asegurar fondo blanco detrás (evita PNG transparente)
            try {
              const vb = svgEl.getAttribute('viewBox');
              let w = parseFloat(svgEl.getAttribute('width'));
              let h = parseFloat(svgEl.getAttribute('height'));
              if ((!w || !h) && vb) {
                const parts = vb.split(/\s+/).map(Number);
                if (parts.length === 4) { w = parts[2]; h = parts[3]; }
              }
              if (w && h) {
                const bg = document.createElementNS('http://www.w3.org/2000/svg','rect');
                bg.setAttribute('x','0');
                bg.setAttribute('y','0');
                bg.setAttribute('width', String(w));
                bg.setAttribute('height', String(h));
                bg.setAttribute('fill', '#ffffff');
                svgEl.insertBefore(bg, svgEl.firstChild);
              }
            } catch {}
            svgRef.current = svgEl;
          }
        }
      } catch (e) {
        setError('No se pudo renderizar el diagrama Mermaid.');
        try { console.error('Mermaid render error', e); } catch {}
      }
    })();
    return () => { cancelled = true; };
  }, [code, idPrefix]);

  const handleExportSvg = () => {
    try {
      const node = svgRef.current;
      if (!node) return;
      const serializer = new XMLSerializer();
      let source = serializer.serializeToString(node);
      if (!source.match(/xmlns=\"http:\/\/www.w3.org\/2000\/svg\"/)) {
        source = source.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
      }
      if (!source.match(/xmlns:xlink=/)) {
        source = source.replace('<svg', '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
      }
      source = '<?xml version="1.0" standalone="no"?>\r\n' + source;
      const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'diagrama.svg';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError('No se pudo exportar el SVG.');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {error && <div style={{ color: '#b91c1c', fontSize: 12 }}>{error}</div>}
      <div ref={containerRef} style={{ overflowX: 'auto', background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 8 }} />
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={handleExportSvg} className="btn btn-yellow" style={{ padding: '8px 12px', border: 'none', borderRadius: 8, fontWeight: 700 }}>
          Exportar SVG
        </button>
      </div>
    </div>
  );
}
  const toggleTheme = () => {
    const next = theme === 'classic' ? 'baufest' : 'classic';
    setTheme(next);
    try { localStorage.setItem('ui_theme', next); } catch {}
    document.documentElement.setAttribute('data-theme', next);
  };

  const handleSendMessage = async (textOverride) => {
    const msg = (textOverride ?? question).trim();
    if (!msg) return;
    
    setLoading(true);
    
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          session_id: sessionId, 
          message: msg
        }),
      });
      
      if (!res.ok) {
        throw new Error(`Error: ${res.status}`);
      }
      
      const data = await res.json();
      
      // Actualizar el ID de sesión si es nuevo
      if (!sessionId) {
        setSessionId(data.session_id);
      }
      
      // Actualizar el historial de chat
      setChatHistory(data.history);
      
      // Actualizar las fuentes
      setSources(data.sources || []);
      
      // Limpiar el campo de pregunta si se envió desde el input o por voz
      setQuestion("");

      if (speakReply && data?.message?.content) {
        await playTTS(data.message.content);
      }
    } catch (error) {
      console.error("Error al consultar:", error);
      // Añadir mensaje de error al chat
      setChatHistory([
        ...chatHistory,
        {
          role: "assistant",
          content: "Error al procesar tu consulta. Por favor, intenta nuevamente.",
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const stopTTS = () => {
    // Abortar descarga si está en curso
    const ctrl = ttsAbortRef.current;
    if (ctrl) { try { ctrl.abort(); } catch {} ttsAbortRef.current = null; }
    // Abortar todos los controladores de chunks en curso
    if (ttsCtrlsRef.current && Array.isArray(ttsCtrlsRef.current)) {
      for (const c of ttsCtrlsRef.current) { try { c.abort(); } catch {} }
    }
    ttsCtrlsRef.current = [];
    ttsQueueRef.current.forEach(url => { try { URL.revokeObjectURL(url); } catch {} });
    ttsQueueRef.current = [];
    // Detener reproducción si ya empezó
    const a = ttsAudioRef.current;
    if (a) {
      try { a.pause(); } catch {}
      try { a.currentTime = 0; } catch {}
      ttsAudioRef.current = null;
    }
    setTtsActive(false);
  };

  // Utilidad: dividir texto en chunks por frases con límite de caracteres
  function splitIntoChunks(text, maxLen = 240) {
    const parts = [];
    const sentences = text
      .split(/(?<=[\.\!\?\u2026])\s+/) // divide en fin de oración
      .map(s => s.trim())
      .filter(Boolean);
    let buf = '';
    for (const s of sentences) {
      if ((buf + ' ' + s).trim().length <= maxLen) {
        buf = (buf ? buf + ' ' : '') + s;
      } else {
        if (buf) parts.push(buf);
        if (s.length <= maxLen) {
          buf = s;
        } else {
          // fallback: cortar crudo si una sola oración es muy larga
          let i = 0;
          while (i < s.length) {
            parts.push(s.slice(i, i + maxLen));
            i += maxLen;
          }
          buf = '';
        }
      }
    }
    if (buf) parts.push(buf);
    return parts;
  }

  // Prefetch de un chunk a URL con AbortController
  async function fetchChunkUrl(chunkText) {
    const ctrl = new AbortController();
    ttsCtrlsRef.current.push(ctrl);
    const resp = await fetch(`${API_BASE}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: chunkText }),
      signal: ctrl.signal
    });
    if (!resp.ok) throw new Error(`TTS ${resp.status}`);
    const blob = await resp.blob();
    return URL.createObjectURL(blob);
  }

  const playTTS = async (text) => {
    try {
      if (!text) return;
      // Detener cualquier reproducción previa
      if (ttsAudioRef.current) {
        try { ttsAudioRef.current.pause(); } catch {}
        try { ttsAudioRef.current.currentTime = 0; } catch {}
      }
      // Limpiar colas y controladores anteriores
      stopTTS();
      setTtsActive(true);

      const chunks = splitIntoChunks(text);
      if (chunks.length === 0) { setTtsActive(false); return; }

      // 1) Obtener y reproducir el primer chunk
      const firstUrl = await fetchChunkUrl(chunks[0]);
      const audio = new Audio(firstUrl);
      ttsAudioRef.current = audio;

      // 2) Prefetch del resto en background
      (async () => {
        for (let i = 1; i < chunks.length; i++) {
          try {
            const url = await fetchChunkUrl(chunks[i]);
            ttsQueueRef.current.push(url);
          } catch (e) {
            // si falla un chunk, continuar con el siguiente
          }
        }
      })();

      audio.onended = async () => {
        // reproducir siguiente si está listo
        const nextUrl = ttsQueueRef.current.shift();
        if (nextUrl) {
          try {
            const a2 = new Audio(nextUrl);
            ttsAudioRef.current = a2;
            a2.onended = audio.onended;
            await a2.play();
          } catch {
            setTtsActive(false);
          }
        } else {
          // esperar brevemente por si el siguiente llega tarde
          const waitMs = 800;
          await new Promise(r => setTimeout(r, waitMs));
          const lateUrl = ttsQueueRef.current.shift();
          if (lateUrl) {
            try {
              const a3 = new Audio(lateUrl);
              ttsAudioRef.current = a3;
              a3.onended = audio.onended;
              await a3.play();
              return;
            } catch {}
          }
          setTtsActive(false);
        }
      };

      await audio.play();
    } catch (e) {
      setTtsActive(false);
    }
  };

  // --- Voz: grabación y flujo STT -> chat ---
  // (sin streaming)

  

  const startRecording = async () => {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // -------- Modo batch (MediaRecorder + /stt) --------
        const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus' : 'audio/webm';
        const mr = new MediaRecorder(stream, { mimeType: mime });
        mediaRecorderRef.current = mr;
        audioChunksRef.current = [];

        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
        };

        mr.onstop = async () => {
          const blob = new Blob(audioChunksRef.current, { type: mime });
          try {
            const form = new FormData();
            form.append('audio', blob, 'input.webm');
            const sttRes = await fetch(`${API_BASE}/stt`, {
              method: 'POST',
              body: form
            });
            if (!sttRes.ok) {
              const t = await sttRes.text();
              throw new Error(t || `STT error ${sttRes.status}`);
            }
            const sttData = await sttRes.json();
            const transcript = sttData?.text || '';
            if (transcript) {
              // Auto-enviar: agregar al chat y enviar consulta
              setChatHistory(prev => ([
                ...prev,
                {
                  role: "user",
                  content: transcript,
                  timestamp: new Date().toISOString()
                }
              ]));
              handleSendMessage(transcript);
              setQuestion('');
            }
          } catch (e) {
            console.error('Error STT:', e);
          }
        };
        mr.start();
        setRecording(true);
    } catch (e) {
      console.error('No se pudo iniciar grabación:', e);
    }
  };

  const stopRecording = () => {
    if (!recording) return;
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      mr.stop();
      mr.stream?.getTracks()?.forEach(t => t.stop());
    }
    setRecording(false);
  };

  const startNewChat = async () => {
    // 1) Detener cualquier audio TTS en curso y limpiar colas
    try { stopTTS(); } catch {}

    // 2) Detener grabación de voz si estuviera activa
    try { stopRecording(); } catch {}

    // 3) Borrar sesión en backend si existe
    try {
      if (sessionId) {
        await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' });
      }
    } catch {}

    // 4) Reset de estado del backend a inicio fresco
    try {
      const resetRes = await fetch(`${API_BASE}/reset`, { method: 'POST' });
      const resetData = resetRes.ok ? await resetRes.json() : null;
      const welcome = resetData?.welcome_text || '¡Bienvenido! Te ayudo a consultar y resumir el contenido del documento sobre tu documento. ¿Qué te gustaría saber?';
      // 5) Reset de estado de UI y datos locales
      setSessionId("");
      setChatHistory([{ role: 'assistant', content: welcome, timestamp: new Date().toISOString() }]);
      setSources([]);
      setQuestion("");
      setShowSources(false);
      setSelectedFile(null);
      try { if (fileInputRef.current) fileInputRef.current.value = ""; } catch {}
      if (speakReply) { try { await playTTS(welcome); } catch {} }
    } catch {
      // Si falla el reset del backend, aún limpiamos el estado local
      setSessionId("");
      setChatHistory([{ role: 'assistant', content: '¡Bienvenido! Te ayudo a consultar y resumir el contenido del documento sobre tu documento. ¿Qué te gustaría saber?', timestamp: new Date().toISOString() }]);
      setSources([]);
      setQuestion("");
      setShowSources(false);
      setSelectedFile(null);
      try { if (fileInputRef.current) fileInputRef.current.value = ""; } catch {}
    }
  };

  const saveSettingsToEnv = async () => {
    try {
      const res = await fetch(`${API_BASE}/save_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          FRAME_EVERY_SECS: frameEverySecs,
          MAX_FRAMES: maxFrames,
          SCENE_DETECTION: sceneDetection,
          DO_OCR_FRAMES: doOcrFrames,
          SHOW_SOURCES: showSources,
          SPEAK_REPLY: speakReply,
          OCR_ENGINE: ocrEngine,
        }),
      });
      if (res.ok) {
        alert('✅ Configuración guardada en .env');
      } else {
        alert('❌ Error al guardar configuración');
      }
    } catch (err) {
      alert('❌ Error: ' + err.message);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      const qp = new URLSearchParams({
        frame_every_secs: String(frameEverySecs ?? 3),
        max_frames: String(maxFrames ?? 200),
        scene_detection: String(!!sceneDetection),
        do_ocr_frames: String(!!doOcrFrames),
        ocr_engine: String(ocrEngine || 'azure'),
      });
      const res = await fetch(`${API_BASE}/upload?${qp.toString()}`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `Error: ${res.status}`);
      }
      // Resetear chat al reindexar (solo en UI; NO resetear backend)
      const data = await res.json();
      try { stopTTS(); } catch {}
      setSessionId("");
      setSources([]);
      setQuestion("");
      setShowSources(false);
      setSelectedFile(null);
      // Mensaje de bienvenida genérico fijo (alineado con backend)
      const welcome = "¡Bienvenido! Te ayudo a consultar sobre la información disponible. ¿Qué te gustaría saber?";
      setChatHistory([{
        role: "assistant",
        content: welcome,
        timestamp: new Date().toISOString()
      }]);

      if (speakReply) {
        await playTTS(welcome);
      }
    } catch (e) {
      console.error(e);
      alert("Error al cargar/indexar el archivo. Revise el backend.");
    } finally {
      setUploading(false);
    }
  };

  const exportChatToTxt = () => {
    if (chatHistory.length === 0) return;
    
    // Crear contenido del archivo
    let content = "Demo de Recuperación Inteligente de Información (videos + docs)\n";
    content += "Fecha: " + new Date().toLocaleDateString() + "\n";
    content += "ID de Sesión: " + sessionId + "\n\n";
    
    // Añadir mensajes
    chatHistory.forEach((msg, index) => {
      content += `[${new Date(msg.timestamp).toLocaleString()}] `;
      content += `${msg.role === 'user' ? 'USUARIO' : 'ASISTENTE'}:\n`;
      content += `${msg.content}\n\n`;
    });
    
    // Crear y descargar el archivo
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `chat_${sessionId.substring(0, 8)}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const exportAssistantToDocx = async (text, filename = 'manual.docx') => {
    try {
      const { Document, Packer, Paragraph, HeadingLevel, TextRun } = await import('docx');

      // Convierte una línea de Markdown simple a una lista de TextRun con formato
      const mdInlineToRuns = (line) => {
        const runs = [];
        if (!line) return runs;
        const regex = /(\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_|`[^`]+`)/g;
        let lastIndex = 0;
        let m;
        while ((m = regex.exec(line)) !== null) {
          if (m.index > lastIndex) {
            runs.push(new TextRun(line.slice(lastIndex, m.index)));
          }
          const token = m[0];
          if (token.startsWith('**') && token.endsWith('**')) {
            runs.push(new TextRun({ text: token.slice(2, -2), bold: true }));
          } else if (token.startsWith('__') && token.endsWith('__')) {
            runs.push(new TextRun({ text: token.slice(2, -2), bold: true }));
          } else if (token.startsWith('*') && token.endsWith('*')) {
            runs.push(new TextRun({ text: token.slice(1, -1), italics: true }));
          } else if (token.startsWith('_') && token.endsWith('_')) {
            runs.push(new TextRun({ text: token.slice(1, -1), italics: true }));
          } else if (token.startsWith('`') && token.endsWith('`')) {
            runs.push(new TextRun({ text: token.slice(1, -1), font: { name: 'Courier New' } }));
          } else {
            runs.push(new TextRun(token));
          }
          lastIndex = regex.lastIndex;
        }
        if (lastIndex < line.length) {
          runs.push(new TextRun(line.slice(lastIndex)));
        }
        return runs;
      };

      const paragraphs = [];
      const lines = String(text || '').split(/\r?\n/);
      for (const raw of lines) {
        const line = raw.trimEnd();
        if (!line.trim()) {
          paragraphs.push(new Paragraph(''));
          continue;
        }
        if (line.startsWith('### ')) {
          paragraphs.push(new Paragraph({ text: line.replace(/^###\s+/, ''), heading: HeadingLevel.HEADING_3 }));
        } else if (line.startsWith('## ')) {
          paragraphs.push(new Paragraph({ text: line.replace(/^##\s+/, ''), heading: HeadingLevel.HEADING_2 }));
        } else if (line.startsWith('# ')) {
          paragraphs.push(new Paragraph({ text: line.replace(/^#\s+/, ''), heading: HeadingLevel.HEADING_1 }));
        } else {
          paragraphs.push(new Paragraph({ children: mdInlineToRuns(line) }));
        }
      }

      const doc = new Document({ sections: [{ properties: {}, children: paragraphs }] });
      const blob = await Packer.toBlob(doc);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      console.error('Error exportando DOCX', e);
      alert('No se pudo exportar el DOCX.');
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--page-bg)", color: "var(--body-text)", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "16px 24px", background: "var(--header-bg)", borderBottom: "1px solid var(--header-border)", display: "flex", justifyContent: "space-between", alignItems: "center", color: "var(--header-fg)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Image src={theme === 'baufest' ? LogoBaufestWhite : LogoBaufest} alt="Baufest" width={120} height={120} style={{ objectFit: "contain" }} />
          <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 700, color: "var(--title-color)", lineHeight: 1.1 }}>
            <span>Demo de Recuperación Inteligente</span><br/>
            <span>de Información (videos + docs)</span>
          </h1>
        </div>
        <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, width: 260 }}>
            <input ref={fileInputRef} type="file" accept=".mp4,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain" onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} style={{ color: "inherit", width: "100%" }} />
            <button onClick={handleUpload} disabled={!selectedFile || uploading} className="btn btn-yellow" style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "none", cursor: (!selectedFile || uploading) ? "not-allowed" : "pointer", fontWeight: 700 }}>{uploading ? "Cargando..." : "Subir e indexar archivo"}</button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, width: 260 }}>
            <button onClick={exportChatToTxt} disabled={chatHistory.length === 0} className="btn btn-yellow" style={{ width: "100%", padding: "10px 16px", border: "none", borderRadius: 8, cursor: chatHistory.length === 0 ? "not-allowed" : "pointer", opacity: chatHistory.length === 0 ? 0.7 : 1, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              Exportar chat
            </button>
            <button onClick={() => { const last = [...chatHistory].reverse().find(m => m.role === 'assistant' && m.content); if (last?.content) exportAssistantToDocx(last.content, `manual_${(sessionId||'').slice(0,8)||'sesion'}.docx`); }} disabled={!chatHistory.some(m => m.role === 'assistant' && m.content)} className="btn btn-yellow" style={{ width: "100%", padding: "10px 16px", border: "none", borderRadius: 8, cursor: chatHistory.some(m => m.role === 'assistant' && m.content) ? "pointer" : "not-allowed", opacity: chatHistory.some(m => m.role === 'assistant' && m.content) ? 1 : 0.7, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }} title="Exportar DOCX (última respuesta)">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M10 13h4"></path><path d="M10 17h4"></path></svg>
              Exportar DOCX (última respuesta)
            </button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, width: 180 }}>
            <button onClick={() => setShowSettings(true)} className="btn btn-config" style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "none", fontWeight: 700 }}>Configuración</button>
            <button onClick={startNewChat} className="btn btn-yellow" style={{ width: "100%", padding: "10px 16px", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700 }}>Nueva conversación</button>
            <button onClick={() => setShowAbout(true)} className="btn btn-about" style={{ width: "100%", padding: "10px 12px", border: "none", borderRadius: 8, fontWeight: 700 }}>Acerca de…</button>
          </div>
        </div>
      </div>
      <div style={{ height: 4, background: "var(--brand-accent)" }} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", maxWidth: 1200, margin: "0 auto", width: "100%", padding: "24px", gap: "24px" }}>
        <div style={{ flex: 1, background: "var(--panel-bg)", borderRadius: 12, boxShadow: "0 4px 12px rgba(0,0,0,0.05)", display: "flex", flexDirection: "column", overflow: "hidden", minHeight: "60vh" }}>
          <div style={{ flex: 1, padding: 24, overflowY: "auto", display: "flex", flexDirection: "column", gap: 16 }}>
            {chatHistory.length === 0 ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", color: "#94a3b8", textAlign: "center", padding: "0 24px" }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                <h3 style={{ marginTop: 16, fontWeight: 500 }}>Inicia una nueva conversación</h3>
                <p>Primero sube un archivo (arriba) y luego haz preguntas sobre su contenido.</p>
              </div>
            ) : (
              chatHistory.slice(-20).map((msg, index) => (
                <div key={index} style={{ display: "flex", flexDirection: "column", alignSelf: msg.role === "user" ? "flex-end" : "flex-start", maxWidth: "80%" }}>
                  <div style={{ background: msg.role === "user" ? "var(--user-bubble-bg)" : "var(--assistant-bubble-bg)", color: msg.role === "user" ? "var(--user-text)" : "var(--assistant-text)", padding: "12px 16px", borderRadius: msg.role === "user" ? "12px 12px 0 12px" : "12px 12px 12px 0", boxShadow: "0 1px 2px rgba(0,0,0,0.05)", fontSize: 16, fontWeight: 400, lineHeight: 1.5 }}>
                    {msg.role === 'assistant' ? (<AssistantContent content={msg.content} msgIndex={index} />) : (msg.content)}
                  </div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 4, alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                    <span style={{ fontSize: 12, color: 'var(--timestamp-text)' }}>{new Date(msg.timestamp).toLocaleTimeString()}</span>
                    {msg.role === 'assistant' && msg.intent && (
                      <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 999, background: 'var(--btn-secondary-bg)', color: 'var(--body-text)', border: '1px solid var(--header-border)', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700 }} title={`Intención: ${msg.intent}`}>{msg.intent === 'rag' ? 'RAG' : 'GEN'}</span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
          <div style={{ borderTop: "1px solid var(--header-border)", padding: 16, display: "flex", gap: 12, background: "var(--panel-bg)" }}>
            <input type="text" placeholder="Escribe tu pregunta..." value={question} onChange={(e) => setQuestion(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()} style={{ flex: 1, padding: "12px 16px", borderRadius: 8, border: "1px solid var(--input-border)", fontSize: 16, outline: "none" }} disabled={loading} />
            <button onClick={recording ? stopRecording : startRecording} title={recording ? 'Detener grabación' : 'Hablar'} className={`btn btn-voice ${recording ? 'recording' : ''}`} style={{ padding: "0 14px", border: "none", borderRadius: 8, fontWeight: 600, fontSize: 16, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
              {recording ? (<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>) : (<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 1v22"/><path d="M8 5a4 4 0 0 1 8 0v6a4 4 0 0 1-8 0Z"/><path d="M5 11a7 7 0 0 0 14 0"/></svg>)}
            </button>
            <button onClick={() => handleSendMessage()} disabled={loading || !question} className="btn btn-yellow" style={{ padding: "0 20px", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 16, cursor: loading || !question ? "not-allowed" : "pointer", opacity: loading || !question ? 0.7 : 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
              {loading ? (<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><circle cx="12" cy="12" r="10" opacity="0.25" /><path d="M12 2a10 10 0 0 1 10 10" /></svg>) : (<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>)}
            </button>
          </div>
        </div>
        {showSources && sources.length > 0 && (
          <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 4px 12px rgba(0,0,0,0.05)", padding: 24 }}>
            <h3 style={{ color: "#2563eb", marginBottom: 16, fontWeight: 600 }}>Fuentes de información:</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {sources.map((source, index) => (
                <div key={index} style={{ background: "#f8fafc", padding: 16, borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 14, boxShadow: "0 1px 3px rgba(0,0,0,0.05)", position: "relative", overflow: "hidden" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontWeight: 600 }}>Fragmento {index + 1}</span>
                    <span style={{ background: "#2563eb", color: "white", padding: "3px 10px", borderRadius: 12, fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", gap: "4px" }}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path></svg>
                      Página {source.page_number}
                    </span>
                  </div>
                  {source.page_image && (
                    <div style={{ marginBottom: 16, borderRadius: 8, overflow: "hidden", boxShadow: "0 2px 8px rgba(0,0,0,0.1)" }}>
                      <img src={`http://localhost:8000${source.page_image}`} alt={`Página ${source.page_number} del PDF`} style={{ width: "100%", display: "block" }} />
                    </div>
                  )}
                  <details style={{ marginTop: 8, color: "#4a5568" }}>
                    <summary style={{ cursor: "pointer", fontWeight: 500 }}>Ver texto extraído</summary>
                    <p style={{ lineHeight: 1.6, whiteSpace: "pre-wrap", textAlign: "justify", padding: "8px 0", fontSize: "13px", color: "#666" }}>{source.text}</p>
                  </details>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      {/* Modal de configuración */}
      {showSettings && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2000 }}>
          <div style={{ background: '#fff', color: '#111', borderRadius: 12, padding: 20, width: 420, maxWidth: '90vw', boxShadow: '0 10px 30px rgba(0,0,0,0.2)' }}>
            <h3 style={{ marginTop: 0, marginBottom: 12 }}>Configuración</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input type="checkbox" checked={showSources} onChange={() => setShowSources(!showSources)} style={{ marginRight: 8 }} />
                Mostrar fuentes
              </label>
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input type="checkbox" checked={speakReply} onChange={() => setSpeakReply(!speakReply)} style={{ marginRight: 8 }} />
                Escuchar respuesta
              </label>
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input type="checkbox" checked={theme === 'baufest'} onChange={toggleTheme} style={{ marginRight: 8 }} />
                Modo oscuro
              </label>
              <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb' }} />
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input type="checkbox" checked={doOcrFrames} onChange={() => setDoOcrFrames(!doOcrFrames)} style={{ marginRight: 8 }} />
                Realizar OCR en frames del video
              </label>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#555', marginBottom: 6 }}>Motor de OCR</label>
                <div style={{ display: 'flex', gap: 12 }}>
                  <label style={{ display: 'flex', alignItems: 'center' }}>
                    <input type="radio" name="ocr-engine" checked={ocrEngine === 'azure'} onChange={() => setOcrEngine('azure')} style={{ marginRight: 8 }} />
                    Azure (GPT‑4o)
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center' }}>
                    <input type="radio" name="ocr-engine" checked={ocrEngine === 'tesseract'} onChange={() => setOcrEngine('tesseract')} style={{ marginRight: 8 }} />
                    Tesseract
                  </label>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 12 }}>
                <div style={{ flex: 1 }}>
                  <label style={{ display: 'block', fontSize: 12, color: '#555' }}>Frames cada... (segundos)</label>
                  <input type="number" min={1} value={frameEverySecs} onChange={(e) => setFrameEverySecs(Math.max(1, Number(e.target.value)||1))} style={{ width: '100%', padding: '8px 10px', border: '1px solid #e5e7eb', borderRadius: 8 }} />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ display: 'block', fontSize: 12, color: '#555' }}>Máx. frames (0 = sin límite)</label>
                  <input type="number" min={0} value={maxFrames} onChange={(e) => setMaxFrames(Math.max(0, Number(e.target.value)||0))} style={{ width: '100%', padding: '8px 10px', border: '1px solid #e5e7eb', borderRadius: 8 }} />
                </div>
              </div>
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input type="checkbox" checked={sceneDetection} onChange={() => setSceneDetection(!sceneDetection)} style={{ marginRight: 8 }} />
                Detección de escenas
              </label>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button onClick={saveSettingsToEnv} className="btn btn-primary" style={{ padding: '8px 12px', border: 'none', borderRadius: 8, fontWeight: 700, marginRight: 'auto' }}>Guardar en .env</button>
              <button onClick={() => setShowSettings(false)} className="btn btn-yellow" style={{ padding: '8px 12px', border: 'none', borderRadius: 8, fontWeight: 700 }}>Cerrar</button>
            </div>
          </div>
        </div>
      )}
      {/* Modal Acerca de */}
      {showAbout && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2000 }}>
          <div style={{ background: '#fff', color: '#111', borderRadius: 12, padding: 20, width: 520, maxWidth: '92vw', boxShadow: '0 10px 30px rgba(0,0,0,0.2)' }}>
            <h3 style={{ marginTop: 0, marginBottom: 12 }}>Acerca de esta demo</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, lineHeight: 1.5 }}>
              <p style={{ margin: 0 }}>Demo de Recuperación Inteligente de Información (videos + documentos) con indexado, extracción de frames y OCR.</p>
              <p style={{ margin: 0 }}>Incluye soporte para transcripción de video, manejo por voz, renderizado de diagramas Mermaid y exportación a DOCX.</p>
              <p style={{ margin: 0 }}>Construido por Baufest como prototipo de capacidades de IA aplicada.</p>
              <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <a href="https://baufest.com/" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', color: '#2563eb', fontWeight: 600 }}>Sitio web</a>
                <Image src={LogoBaufest} alt="Baufest" style={{ height: 'auto', width: 96 }} />
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button onClick={() => setShowAbout(false)} className="btn btn-yellow" style={{ padding: '8px 12px', border: 'none', borderRadius: 8, fontWeight: 700 }}>Cerrar</button>
            </div>
          </div>
        </div>
      )}
      <style jsx global>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        body { margin: 0; padding: 0; }
        :root {
          --page-bg: #f7fafc; --body-text: #222; --header-bg: #fff; --header-border: #e2e8f0; --header-fg: #222; --title-color: #2563eb; --panel-bg: #fff; --btn-primary-bg: #2563eb; --btn-primary-text: #fff; --btn-secondary-bg: #e2e8f0; --assistant-bubble-bg: #f1f5f9; --assistant-text: #1e293b; --user-bubble-bg: #2563eb; --user-text: #fff; --timestamp-text: #94a3b8; --input-border: #e2e8f0; --btn-voice-bg: #10b981;
        }
        [data-theme="baufest"] {
          --page-bg: #f3f4f6; --body-text: #111111; --header-bg: #000000; --header-border: #1f2937; --header-fg: #ffffff; --title-color: #ffffff; --panel-bg: #ffffff; --btn-primary-bg: #ffe11a; --btn-primary-text: #111111; --btn-secondary-bg: #f3f4f6; --assistant-bubble-bg: #f5f5f5; --assistant-text: #111111; --user-bubble-bg: #000000; --user-text: #ffffff; --timestamp-text: #6b7280; --input-border: #e5e7eb; --btn-voice-bg: #111111;
        }
        .assistant-msg p { margin: 0 0 8px; }
        .assistant-msg ul, .assistant-msg ol { margin: 8px 0 8px 20px; }
        .assistant-msg li { margin: 4px 0; }
        .assistant-msg strong { font-weight: 700; }
        .btn-config { background-color: #FF3AAC; color: #ffffff; }
        .btn-config:hover { background-color: #e62f96; }
        .btn-about { background: var(--btn-secondary-bg); color: var(--body-text); }
        .btn-about:hover { background: #e5e7eb; }
      `}</style>
      {speakReply && ttsActive && (
        <div style={{ position: "fixed", right: 24, bottom: 24, zIndex: 1000 }}>
          <button onClick={stopTTS} title="Detener TTS" style={{ padding: "10px 14px", borderRadius: 24, border: "none", background: "var(--brand-fuchsia)", color: "#ffffff", cursor: "pointer", fontWeight: 700, boxShadow: "0 6px 16px rgba(0,0,0,0.15)" }}>Detener voz</button>
        </div>
      )}
    </div>
  );
}

// --- Minimal Markdown renderer (bold, italics, lists, paragraphs) ---
function renderMarkdown(md) {
  if (!md) return { __html: '' };
  try {
    let s = md;
    // Normalize list markers that may come inline
    s = s.replace(/\s+(\d+\.)\s/g, '\n$1 ');
    s = s.replace(/\s+([*\-])\s/g, '\n$1 ');
    // Escape basic HTML
    s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    // Inline code first to protect contents
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Headings (support up to ###)
    s = s.replace(/^###\s+(.+)$/gm, '<h3>$1<\/h3>');
    s = s.replace(/^##\s+(.+)$/gm, '<h2>$1<\/h2>');
    s = s.replace(/^#\s+(.+)$/gm, '<h1>$1<\/h1>');
    // Bold (**, __) then italics (*, _), non-greedy
    s = s.replace(/(\*\*|__)(.+?)\1/g, '<strong>$2</strong>');
    s = s.replace(/(^|[^*])\*(?!\*)([^*]+)\*(?!\*)/g, '$1<em>$2</em>');
    s = s.replace(/(^|[^_])_(?!_)([^_]+)_(?!_)/g, '$1<em>$2</em>');

    // Build HTML by lines, handling lists and headings without wrapping them in <p>
    const lines = s.split(/\r?\n/);
    let html = '';
    let inOl = false, inUl = false;
    const closeLists = () => {
      if (inOl) { html += '</ol>'; inOl = false; }
      if (inUl) { html += '</ul>'; inUl = false; }
    };
    for (let line of lines) {
      if (/^<h[1-3]>/.test(line.trim())) {
        closeLists();
        html += line.trim();
        continue;
      }
      const olMatch = line.match(/^\s*(\d+)\.\s+(.*)$/);
      const ulMatch = line.match(/^\s*[-*]\s+(.*)$/);
      if (olMatch) {
        if (inUl) { html += '</ul>'; inUl = false; }
        if (!inOl) { html += '<ol class="assistant-msg">'; inOl = true; }
        html += `<li>${olMatch[2]}</li>`;
        continue;
      }
      if (ulMatch) {
        if (inOl) { html += '</ol>'; inOl = false; }
        if (!inUl) { html += '<ul class="assistant-msg">'; inUl = true; }
        html += `<li>${ulMatch[1]}</li>`;
        continue;
      }
      if (line.trim() !== '') {
        closeLists();
        html += `<p class="assistant-msg">${line}</p>`;
      }
    }
    closeLists();
    return { __html: html };
  } catch {
    return { __html: md };
  }
}

// ... (rest of the code remains the same)
