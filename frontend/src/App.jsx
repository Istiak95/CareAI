import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const TOKEN_STORAGE_KEY = "medinlp_auth_token";

const ALL_SYMPTOMS = [
  "fever", "cough", "headache", "fatigue", "chest pain", "nausea", "sore throat", "body ache",
  "dizziness", "vomiting", "diarrhea", "shortness of breath", "abdominal pain", "back pain",
  "joint pain", "muscle pain", "sore eyes", "runny nose", "sneezing", "sweating", "chills",
  "rash", "itching", "swelling", "difficulty breathing", "loss of appetite", "weight loss",
  "anxiety", "depression", "insomnia", "palpitation", "blood in urine", "blurred vision"
];

const MEDICAL_VOICE_TERMS = [
  "fever", "cough", "headache", "chest pain", "shortness of breath", "breathing problem", "vomiting",
  "dizziness", "diarrhea", "sore throat", "fatigue", "rash", "itching", "nausea", "palpitation",
  "stomach pain", "abdominal pain", "back pain", "joint pain",
  "kashi", "khasi", "jor", "jhor", "matha betha", "matha batha", "buk betha", "buke batha", "shash kosto",
  "জ্বর", "কাশি", "মাথা ব্যথা", "বুক ব্যথা", "শ্বাস কষ্ট", "পেট ব্যথা", "বমি"
];

const TRANSCRIPT_FIXES = [
  [/\bfiver\b/gi, "fever"],
  [/\bfeaver\b/gi, "fever"],
  [/\bfavour\b/gi, "fever"],
  [/\bjori?e?\b/gi, "jor"],
  [/\bjhor\b/gi, "jor"],
  [/\bzhor\b/gi, "jor"],
  [/\bkhasi\b/gi, "kashi"],
  [/\bcasi\b/gi, "kashi"],
  [/\bmath\s+batha\b/gi, "matha betha"],
  [/\bmatha\s+batha\b/gi, "matha betha"],
  [/\bbuke?\s+batha\b/gi, "buk betha"],
  [/\bbukhe\s+batha\b/gi, "buk betha"],
  [/\bsas\s+nite\s+(problem|prblm|kosto)\b/gi, "shash nite problem"],
  [/\bshortness breath\b/gi, "shortness of breath"],
  [/\bshortness off breath\b/gi, "shortness of breath"],
  [/\bchest pin\b/gi, "chest pain"],
  [/\bhead ache\b/gi, "headache"],
  [/\bstomach ache\b/gi, "stomach pain"],
  [/আমার/gi, "আমার"],
];

function getTimestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function createWelcomeMessage() {
  return {
    id: 0,
    role: "bot",
    content: "Hello, I'm CareAI 👋\n\nDescribe your symptoms in English, Banglish, or Bangla. Examples: fever and cough / amar jor ase kashi hocche / chest pain",
    timestamp: getTimestamp()
  };
}

function makeChatTitle(text) {
  const cleaned = String(text || "New chat").replace(/\s+/g, " ").trim();
  if (!cleaned) return "New chat";
  return cleaned.length > 48 ? `${cleaned.slice(0, 48)}...` : cleaned;
}

function normalizeTranscript(text) {
  let value = String(text || "")
    .replace(/[।!?.,;:]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  TRANSCRIPT_FIXES.forEach(([pattern, replacement]) => {
    value = value.replace(pattern, replacement);
  });

  return value.replace(/\s+/g, " ").trim();
}

function pickBestTranscript(result) {
  const alternatives = Array.from({ length: result.length }, (_, index) => result[index]).filter(Boolean);
  if (!alternatives.length) return "";

  let bestText = alternatives[0].transcript || "";
  let bestScore = -1;

  alternatives.forEach((alternative) => {
    const transcript = normalizeTranscript(alternative.transcript || "");
    const lower = transcript.toLowerCase();
    const medicalHits = MEDICAL_VOICE_TERMS.reduce((count, term) => count + (lower.includes(term) ? 1 : 0), 0);
    const confidence = typeof alternative.confidence === "number" ? alternative.confidence : 0;
    const score = confidence + medicalHits * 0.12;

    if (score > bestScore) {
      bestScore = score;
      bestText = transcript;
    }
  });

  return bestText;
}

// ============================================================
// Specialist Finder - Text-based Google Maps search
// ============================================================
function getMapSearchTerm(recommendation, diseaseName) {
  const disease = String(diseaseName || recommendation?.disease || "").toLowerCase();

  const specialistMap = {
    "asthma": "Chest Specialist",
    "diabetes": "Diabetes Specialist",
    "heart": "Cardiologist",
    "heart disease": "Cardiologist",
    "hypertension": "Cardiologist",
    "skin": "Dermatologist",
    "kidney": "Nephrologist",
    "infection": "Medicine Specialist",
    "fever": "Medicine Specialist",
    "dengue": "Medicine Specialist",
    "malaria": "Medicine Specialist",
    "pneumonia": "Chest Specialist",
    "tuberculosis": "Chest Specialist",
    "covid": "Medicine Specialist",
    "migraine": "Neurologist",
    "stroke": "Neurologist",
    "depression": "Psychiatrist",
    "anxiety": "Psychiatrist",
    "pregnancy": "Gynecologist",
    "urinary": "Urologist",
    "gastritis": "Gastroenterologist",
    "liver": "Gastroenterologist"
  };

  for (const key in specialistMap) {
    if (disease.includes(key)) return specialistMap[key];
  }

  const possibleDoctor =
    recommendation?.doctor_type_patient_should_see ||
    recommendation?.doctor_type ||
    recommendation?.doctor ||
    recommendation?.specialist ||
    recommendation?.specialist_type;

  if (possibleDoctor && typeof possibleDoctor === "string") {
    let term = possibleDoctor.trim().split(";")[0].split(",")[0].trim();
    const lower = term.toLowerCase();

    if (
      lower.includes("emergency") ||
      lower.includes("if severe") ||
      lower.includes("safety") ||
      lower.includes("seek urgent") ||
      lower.includes("call") ||
      term.length > 35
    ) {
      return "hospital";
    }

    if (lower === "primary care") return "Medicine Specialist";
    return term || "hospital";
  }

  return "hospital";
}

function openNearbyMap(recommendation, diseaseName) {
  const searchTerm = getMapSearchTerm(recommendation, diseaseName);
  const query = `${searchTerm} hospital near me`;
  const mapUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;
  window.open(mapUrl, "_blank", "noopener,noreferrer");
}

function SpecialistFinder({ recommendation, diseaseName }) {
  const specialist = getMapSearchTerm(recommendation, diseaseName);
  return (
    <div className="specialist-finder">
      <button
        onClick={() => openNearbyMap(recommendation, diseaseName)}
        className="specialist-btn"
        title={`Find nearby ${specialist}`}
      >
        🏥 Find Nearby {specialist}
      </button>
    </div>
  );
}

function Tooltip({ text, children }) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="tooltip-wrapper" onMouseEnter={() => setVisible(true)} onMouseLeave={() => setVisible(false)}>
      {children}
      {visible && <div className="tooltip">{text}</div>}
    </div>
  );
}

function SymptomContributionChart({ shapItems }) {
  const topSymptoms = [...shapItems]
    .sort((a, b) => Math.abs(parseFloat(b.contribution)) - Math.abs(parseFloat(a.contribution)))
    .slice(0, 3);

  const maxValue = Math.max(...topSymptoms.map((s) => Math.abs(parseFloat(s.contribution))), 1);

  return (
    <div className="contribution-chart">
      {topSymptoms.map((item, i) => {
        const value = parseFloat(item.contribution);
        const isPositive = value > 0;
        const percentage = (Math.abs(value) / maxValue) * 100;
        return (
          <div key={i} className="chart-item">
            <span className="chart-label">{item.symptom}</span>
            <div className="chart-bar-container">
              <div className={`chart-bar ${isPositive ? "positive" : "negative"}`} style={{ width: `${percentage}%` }} />
            </div>
            <span className={`chart-importance ${isPositive ? "important" : "not-important"}`}>
              {isPositive ? "important key symptom" : "not important key symptom"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function PredictionCard({ pred, isPrimary, index }) {
  const rec = pred.recommendation || {};
  const shapItems = pred.shap_explanation?.present_symptom_contributions || [];
  const confidence = pred.confidence_percent || 0;

  const getConfidenceEmoji = (conf) => {
    if (conf >= 70) return "🟢";
    if (conf >= 50) return "🟡";
    return "🔴";
  };

  const getConfidenceLabel = (conf) => {
    if (conf >= 70) return "High";
    if (conf >= 50) return "Medium";
    return "Low";
  };

  return (
    <div className={`prediction-card ${isPrimary ? "primary" : ""}`} style={{ animationDelay: `${index * 0.1}s` }}>
      <div className="card-header">
        <div className="rank-badge">#{pred.rank}</div>
        <h3>{pred.disease}</h3>
        <div className="confidence-badge">
          <span className="conf-emoji">{getConfidenceEmoji(confidence)}</span>
          <span>{confidence}%</span>
          <Tooltip text={getConfidenceLabel(confidence) === "High" ? "Very likely condition" : getConfidenceLabel(confidence) === "Medium" ? "Moderate likelihood" : "Less likely but possible"}>
            <span className="info-icon">ℹ</span>
          </Tooltip>
        </div>
      </div>

      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{
            width: `${confidence}%`,
            background: confidence > 70 ? "#10b981" : confidence > 50 ? "#f59e0b" : "#3b82f6"
          }}
        />
      </div>

      <div className="confidence-level">{getConfidenceLabel(confidence)} Confidence</div>

      {rec.doctor_type_patient_should_see && (
        <div className="info-row">
          <span className="label">See:</span>
          <span className="value">{rec.doctor_type_patient_should_see}</span>
        </div>
      )}

      {rec.common_tests_to_discuss_with_clinician?.length > 0 && (
        <div className="info-row">
          <span className="label">Tests:</span>
          <span className="value">{rec.common_tests_to_discuss_with_clinician.join(", ")}</span>
        </div>
      )}

      {rec.short_care_note && (
        <div className="info-row">
          <span className="label">Note:</span>
          <span className="value">{rec.short_care_note}</span>
        </div>
      )}

      {rec.urgency_level && (
        <div className="info-row">
          <span className="label">Urgency:</span>
          <span className="value">{rec.urgency_level}</span>
        </div>
      )}

      <SpecialistFinder recommendation={rec} diseaseName={pred.disease} />

      {shapItems.length > 0 && (
        <div className="chart-section">
          <div className="chart-title">Symptoms Importance</div>
          <SymptomContributionChart shapItems={shapItems} />
        </div>
      )}
    </div>
  );
}

function ChatBubble({ role, children, timestamp }) {
  return (
    <div className={`message-wrapper ${role}`}>
      <div className={`avatar avatar-${role}`}>{role === "bot" ? "C" : "You"}</div>
      <div className="bubble-container">
        <div className={`bubble ${role}`}>{children}</div>
        {timestamp && <div className="timestamp">{timestamp}</div>}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message-wrapper bot">
      <div className="avatar avatar-bot">M</div>
      <div className="bubble-container">
        <div className="bubble bot typing-bubble">
          <span className="typing-dot"></span>
          <span className="typing-dot"></span>
          <span className="typing-dot"></span>
        </div>
      </div>
    </div>
  );
}

function MicIcon({ listening }) {
  return (
    <span className="mic-gpt-icon" aria-hidden="true">
      {listening ? (
        <span className="mic-stop-square" />
      ) : (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <path d="M12 19v3" />
        </svg>
      )}
    </span>
  );
}

function AuthPanel({ user,authLoading, authMode, setAuthMode, authForm, setAuthForm, authError, onSubmit, onLogout }) {
  if (user) {
    return (
      <div className="auth-section signed-in">
        <div className="auth-user-row">
          <div className="auth-avatar">{String(user.name || user.email || "U").slice(0, 1).toUpperCase()}</div>
          <div>
            <strong>{user.name || "MediNLP User"}</strong>
            <span>{user.email}</span>
          </div>
        </div>
        <button className="auth-secondary-btn" onClick={onLogout}>Logout</button>
      </div>
    );
  }

  return (
    <div className="auth-section">
      <div className="auth-tabs">
        <button className={authMode === "login" ? "active" : ""} onClick={() => setAuthMode("login")}>Login</button>
        <button className={authMode === "register" ? "active" : ""} onClick={() => setAuthMode("register")}>Register</button>
      </div>

      <form className="auth-form" onSubmit={onSubmit}>
        {authMode === "register" && (
          <div className="auth-name-grid">
            <input
              value={authForm.first_name}
              onChange={(e) => setAuthForm((prev) => ({ ...prev, first_name: e.target.value }))}
              placeholder="First name"
            />
            <input
              value={authForm.last_name}
              onChange={(e) => setAuthForm((prev) => ({ ...prev, last_name: e.target.value }))}
              placeholder="Last name"
            />
          </div>
        )}
        <input
          value={authForm.email}
          onChange={(e) => setAuthForm((prev) => ({ ...prev, email: e.target.value }))}
          placeholder="Email"
          type="email"
        />
        <input
          value={authForm.password}
          onChange={(e) => setAuthForm((prev) => ({ ...prev, password: e.target.value }))}
          placeholder="Password"
          type="password"
        />
        
      {authError && <div className="auth-error">{authError}</div>}

<button
  className="auth-primary-btn auth-submit-button"
  type="submit"
  disabled={authLoading}
>
  {authLoading ? (
    <>
      <span className="login-spinner"></span>
      <span>
        {authMode === "login"
          ? "Logging in..."
          : "Creating account..."}
      </span>
    </>
  ) : (
    <span>
      {authMode === "login"
        ? "Login"
        : "Create account"}
    </span>
  )}
</button>

</form>

      <p className="guest-copy">Guest mode is available, but chat restore/history works only after login.</p>
    </div>
  );
}

function AuthGate({ authMode, setAuthMode, authForm, setAuthForm, authError, authLoading, onSubmit, onGuest, darkMode, setDarkMode }) {
  return (
    <div className={`auth-gate ${darkMode ? "dark-mode" : ""}`}>
      <button
        className="auth-gate-theme-toggle"
        onClick={() => setDarkMode(!darkMode)}
        title={`Switch to ${darkMode ? "light" : "dark"} mode`}
        aria-label="Toggle theme"
      >
        {darkMode ? "☀️" : "🌙"}
      </button>
      <div className="auth-gate-card">
        <div className="auth-gate-logo">C</div>
        <h1>Welcome to CareAI</h1>
        <p className="auth-gate-subtitle">Login to restore previous chats, or continue as guest for a temporary session.</p>

        <AuthPanel
          user={null}
          authMode={authMode}
          setAuthMode={setAuthMode}
          authForm={authForm}
          setAuthForm={setAuthForm}
          authError={authError}
          onSubmit={onSubmit}
          onLogout={() => {}}
        />

        <div className="auth-gate-divider"><span>or</span></div>
        <button className="guest-start-btn" type="button" onClick={onGuest}>
          Continue as Guest
        </button>
        <p className="auth-gate-note">Guest chats are not restored after refresh or logout.</p>
      </div>
    </div>
  );
}

function ChatHistory({ chats, currentChatId, onOpenChat, onDeleteChat, historyLoading }) {
  return (
    <div className="history-section">
      <div className="history-header">
        <h3>Chat History</h3>
      </div>
      {historyLoading && <div className="history-empty">Loading...</div>}
      {!historyLoading && chats.length === 0 && <div className="history-empty">No saved chats yet.</div>}
      <div className="history-list">
        {chats.map((chat) => (
          <button
            key={chat.id}
            className={`history-item ${currentChatId === chat.id ? "active" : ""}`}
            onClick={() => onOpenChat(chat.id)}
          >
            <span>{chat.title || "New chat"}</span>
            <small>{chat.updated_at ? new Date(chat.updated_at).toLocaleDateString() : ""}</small>
            <b onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id); }} title="Delete chat">×</b>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([createWelcomeMessage()]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);  
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState("");
  const [voiceLanguage, setVoiceLanguage] = useState("bn-BD");

  const [authToken, setAuthToken] = useState(() => localStorage.getItem(TOKEN_STORAGE_KEY) || "");
  const [authChecking, setAuthChecking] = useState(() => Boolean(localStorage.getItem(TOKEN_STORAGE_KEY)));
  const [guestMode, setGuestMode] = useState(false);
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({ first_name: "", last_name: "", email: "", password: "" });
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [autoSaveStatus, setAutoSaveStatus] = useState("");

  const messagesEndRef = useRef(null);
  const messageCountRef = useRef(1);
  const resultsRef = useRef(null);
  const recognitionRef = useRef(null);
  const voiceSeedRef = useRef("");

  const getAuthHeaders = (tokenOverride) => {
    const token = tokenOverride || authToken;
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const sanitizeMessagesForStorage = (items) => {
    return items.map((item) => ({
      id: item.id,
      role: item.role,
      content: item.content,
      type: item.type,
      data: item.data,
      timestamp: item.timestamp
    }));
  };

  const handleInputChange = (value) => {
    setInput(value);
    const parts = value.split(",").map((p) => p.trim());
    const lastPart = parts[parts.length - 1].toLowerCase();

    if (lastPart.length > 0) {
      const filtered = ALL_SYMPTOMS.filter((s) => s.toLowerCase().includes(lastPart));
      setSuggestions(filtered);
      setShowSuggestions(true);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  };

  const addSuggestion = (symptom) => {
    const parts = input.split(",").map((p) => p.trim());
    parts[parts.length - 1] = symptom;
    const newInput = parts.join(", ") + ", ";
    setInput(newInput);
    setSuggestions([]);
    setShowSuggestions(false);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
  if (!authToken) {
    setAuthChecking(false);
    return;
  }

  // নতুন করে login করার সময় user আগে থেকেই set থাকলে
  // আবার /api/auth/me request পাঠাবে না
  if (user) {
    setAuthChecking(false);
    return;
  }

  let cancelled = false;

  async function restoreUser() {
    try {
      const res = await fetch(
        `${API_BASE}/api/auth/me`,
        {
          headers: getAuthHeaders(authToken)
        }
      );

      if (!res.ok) {
        throw new Error("Session expired");
      }

      const data = await res.json();

      if (cancelled) return;

      setUser(data.user);
      setGuestMode(false);
      setAuthChecking(false);

      // Chat history background-এ load হবে
      void loadChatList(authToken);
    } catch (err) {
      if (cancelled) return;

      localStorage.removeItem(TOKEN_STORAGE_KEY);
      setAuthToken("");
      setUser(null);
      setAuthChecking(false);
    }
  }

  restoreUser();

  return () => {
    cancelled = true;
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [authToken, user]);useEffect(() => {
  if (!authToken) {
    setAuthChecking(false);
    return;
  }

  // নতুন করে login করার সময় user আগে থেকেই set থাকলে
  // আবার /api/auth/me request পাঠাবে না
  if (user) {
    setAuthChecking(false);
    return;
  }

  let cancelled = false;

  async function restoreUser() {
    try {
      const res = await fetch(
        `${API_BASE}/api/auth/me`,
        {
          headers: getAuthHeaders(authToken)
        }
      );

      if (!res.ok) {
        throw new Error("Session expired");
      }

      const data = await res.json();

      if (cancelled) return;

      setUser(data.user);
      setGuestMode(false);
      setAuthChecking(false);

      // Chat history background-এ load হবে
      void loadChatList(authToken);
    } catch (err) {
      if (cancelled) return;

      localStorage.removeItem(TOKEN_STORAGE_KEY);
      setAuthToken("");
      setUser(null);
      setAuthChecking(false);
    }
  }

  restoreUser();

  return () => {
    cancelled = true;
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [authToken, user]);

  async function loadChatList(tokenOverride) {
    const token = tokenOverride || authToken;
    if (!token) return;
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/chats`, { headers: getAuthHeaders(token) });
      if (!res.ok) throw new Error("Could not load chats");
      const data = await res.json();
      setChats(data.chats || []);
    } catch (err) {
      console.error(err);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function saveChatToServer(messagesToSave, chatIdOverride = currentChatId, titleSource = "", tokenOverride = authToken) {
    const token = tokenOverride || authToken;
    if (!token) return null;

    const firstUserMessage = messagesToSave.find((msg) => msg.role === "user" && msg.content);
    const title = makeChatTitle(titleSource || firstUserMessage?.content || "New chat");
    const payload = { title, messages: sanitizeMessagesForStorage(messagesToSave) };
    const method = chatIdOverride ? "PUT" : "POST";
    const url = chatIdOverride ? `${API_BASE}/api/chats/${chatIdOverride}` : `${API_BASE}/api/chats`;

    try {
      setAutoSaveStatus("Saving...");
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json", ...getAuthHeaders(token) },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error("Could not save chat");
      const data = await res.json();
      const savedId = data.chat?.id || chatIdOverride;
      setCurrentChatId(savedId);
      setAutoSaveStatus("Saved");

      if (data.chat) {
      setChats((previousChats) => {
      const remainingChats = previousChats.filter(
      (chat) => chat.id !== data.chat.id
      );

      return [data.chat, ...remainingChats];
    });
   }

    return savedId;
    } catch (err) {
      console.error(err);
      setAutoSaveStatus("Save failed");
      return chatIdOverride || null;
    }
  }

  async function saveReportToServer(result, chatId, tokenOverride = authToken) {
    const token = tokenOverride || authToken;
    if (!token || !result) return;
    try {
      await fetch(`${API_BASE}/api/reports`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders(token) },
        body: JSON.stringify({ chat_id: chatId || null, result })
      });
    } catch (err) {
      console.error("Report save failed", err);
    }
  }

  async function handleAuthSubmit(event) {
  event.preventDefault();

  if (authLoading) return;

  setAuthError("");
  setAuthLoading(true);

  const endpoint =
    authMode === "register"
      ? "/api/auth/register"
      : "/api/auth/login";

  const payload =
    authMode === "register"
      ? authForm
      : {
          email: authForm.email,
          password: authForm.password,
        };

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Authentication failed");
    }

    localStorage.setItem(TOKEN_STORAGE_KEY, data.token);

    setAuthToken(data.token);
    setUser(data.user);
    setGuestMode(false);
    setAuthChecking(false);

    setAuthForm({
      first_name: "",
      last_name: "",
      email: "",
      password: "",
    });

    setAuthError("");

    void loadChatList(data.token);

    if (messages.some((msg) => msg.role === "user")) {
      void saveChatToServer(messages, null, "", data.token);
    }
  } catch (err) {
    setAuthError(err.message || "Authentication failed");
  } finally {
    setAuthLoading(false);
  }
}
  function logout() {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setAuthToken("");
    setUser(null);
    setGuestMode(false);
    setChats([]);
    setCurrentChatId(null);
    setAutoSaveStatus("");
    setMessages([createWelcomeMessage()]);
    messageCountRef.current = 1;
  }

  function startNewChat() {
  setCurrentChatId(null);
  setMessages([createWelcomeMessage()]);
  setInput("");
  setShowSuggestions(false);
  setMobileSidebarOpen(false);
  messageCountRef.current = 1;
}

  async function openSavedChat(chatId) {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_BASE}/api/chats/${chatId}`, { headers: getAuthHeaders() });
      if (!res.ok) throw new Error("Could not open chat");
      const data = await res.json();
      const restoredMessages = data.chat?.messages?.length ? data.chat.messages : [createWelcomeMessage()];
      setMessages(restoredMessages);
      setCurrentChatId(data.chat.id);
      setMobileSidebarOpen(false);
      const maxId = Math.max(0, ...restoredMessages.map((msg) => Number(msg.id) || 0));
      messageCountRef.current = maxId + 1;
    } catch (err) {
      console.error(err);
    }
  }

  async function deleteSavedChat(chatId) {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_BASE}/api/chats/${chatId}`, {
        method: "DELETE",
        headers: getAuthHeaders()
      });
      if (!res.ok) throw new Error("Could not delete chat");
      if (currentChatId === chatId) startNewChat();
      await loadChatList();
    } catch (err) {
      console.error(err);
    }
  }

  const startVoiceInput = async () => {
  const SpeechRecognitionAPI =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognitionAPI) {
    setVoiceError(
      "Voice recognition is not supported in this browser. Please try Safari, Chrome, or Edge."
    );
    return;
  }

  if (!window.isSecureContext) {
    setVoiceError("Voice input requires a secure HTTPS connection.");
    return;
  }

  // Stop current recognition
  if (isListening && recognitionRef.current) {
    try {
      recognitionRef.current.abort();
    } catch (error) {
      console.error("Could not stop recognition:", error);
    }

    recognitionRef.current = null;
    setIsListening(false);
    return;
  }

  // Explicitly request microphone permission
  try {
    if (navigator.mediaDevices?.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true
      });

      stream.getTracks().forEach((track) => track.stop());
    }
  } catch (error) {
    console.error("Microphone permission error:", error);

    setVoiceError(
      "Microphone access is blocked. Allow microphone permission for this website in Safari settings."
    );
    return;
  }

  const recognition = new SpeechRecognitionAPI();

  recognitionRef.current = recognition;
  voiceSeedRef.current = input.trim();

  recognition.lang = voiceLanguage;
  recognition.interimResults = false;
  recognition.continuous = false;

  // More stable on mobile Safari
  recognition.maxAlternatives = 1;

  let receivedResult = false;

  setVoiceError("");
  setIsListening(true);

  recognition.onstart = () => {
    setVoiceError("");
    setIsListening(true);
  };

  recognition.onaudiostart = () => {
    console.log("Microphone audio capture started");
  };

  recognition.onspeechstart = () => {
    console.log("Speech detected");
  };

  recognition.onresult = (event) => {
    receivedResult = true;

    const result =
      event.results[event.resultIndex] ||
      event.results[event.results.length - 1];

    const transcript = result?.[0]?.transcript || "";

    const combined = normalizeTranscript(
      `${voiceSeedRef.current} ${transcript}`
    );

    if (combined) {
      setInput(combined);
      setVoiceError("");
    } else {
      setVoiceError("No recognizable speech was received.");
    }
  };

  recognition.onspeechend = () => {
    // Small delay prevents Safari from cutting the final word
    window.setTimeout(() => {
      try {
        recognition.stop();
      } catch (error) {
        console.error("Recognition stop error:", error);
      }
    }, 350);
  };

  recognition.onnomatch = () => {
    setVoiceError(
      "Speech was heard but could not be understood. Speak slowly and try again."
    );
  };

  recognition.onerror = (event) => {
    console.error(
      "Speech recognition error:",
      event.error,
      event.message || ""
    );

    const errorMessages = {
      "no-speech":
        "No speech was detected. Tap the microphone, wait one second, then speak.",
      "audio-capture":
        "The microphone could not capture audio. Check microphone permission in your browser settings.",
      "not-allowed":
        "Microphone permission is blocked. Allow microphone access for this website.",
      "service-not-allowed":
        "This browser did not allow the speech-recognition service. On iPhone, try Safari or use keyboard voice typing.",
        "network":
        "Speech recognition network error. Check your internet connection and try again.",
      "language-not-supported":
        `The selected language (${voiceLanguage}) is not supported by this browser.`,
      "bad-grammar":
        "The speech-recognition language configuration is not supported."
    };

    if (event.error !== "aborted") {
      setVoiceError(
        errorMessages[event.error] ||
          `Voice recognition failed: ${event.error || "unknown error"}`
      );
    }

    setIsListening(false);
  };

  recognition.onend = () => {
    console.log("Speech recognition ended");

    recognitionRef.current = null;
    setIsListening(false);

    if (!receivedResult) {
      console.log("Recognition ended without a transcript");
    }
  };

  try {
    recognition.start();
  } catch (error) {
    console.error("Recognition start error:", error);

    recognitionRef.current = null;
    setIsListening(false);
    setVoiceError(
      "Voice recognition could not start. Wait a moment and tap the microphone again."
    );
  }
};
  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    const newUserMessage = {
      id: messageCountRef.current++,
      role: "user",
      content: text,
      timestamp: getTimestamp()
    };

    const messagesWithUser = [...messages, newUserMessage];
    setMessages(messagesWithUser);
    setInput("");
    setShowSuggestions(false);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, top_k: 3, enable_shap: true, shap_nsamples: 30 })
      });
      const data = await res.json();

      const newBotMessage = {
        id: messageCountRef.current++,
        role: "bot",
        type: "result",
        data,
        timestamp: getTimestamp()
      };
      const finalMessages = [...messagesWithUser, newBotMessage];

    setMessages(finalMessages);

    // Chat এবং report background-এ save হবে
    void (async () => {
    const savedChatId = await saveChatToServer(
      finalMessages,
      currentChatId,
      text
    );

    await saveReportToServer(
      data,
      savedChatId
    );
  })();
    } catch (err) {
      const errorMessage = {
        id: messageCountRef.current++,
        role: "bot",
        content: "❌ Backend connection failed. Make sure FastAPI is running on http://127.0.0.1:8000",
        timestamp: getTimestamp()
      };
      const finalMessages = [...messagesWithUser, errorMessage];
     
     setMessages(finalMessages);

void saveChatToServer(
  finalMessages,
  currentChatId,
  text
);
    } finally {
      setLoading(false);
    }
  }

  function renderBotResult(data) {
    if (data.status === "failed") {
      return (
        <div className="error-box">
          <p>❌ {data.message}</p>
          {data.possible_symptoms?.length > 0 && (
            <div className="possible-symptoms">
              <strong>Possible matches:</strong>
              {data.possible_symptoms.map((item) => (
                <span key={`${item.symptom}-${item.matched_text}`} className="symptom-chip">
                  {item.symptom}
                </span>
              ))}
            </div>
          )}
        </div>
      );
    }

    if (data.status === "red_flag") {
      const rf = data.red_flag_result || {};
      const triggeredSymptoms = rf.triggered_symptoms?.length
        ? rf.triggered_symptoms
        : [...(rf.critical_symptoms || []), ...(rf.major_symptoms || [])];

      return (
        <div className="red-flag">
          <h3>⚠️ Red Flag Alert</h3>
          <p><strong>Severity:</strong> {rf.severity}</p>
          <p><strong>Reason:</strong> {rf.reason}</p>

          {triggeredSymptoms.length > 0 && (
            <div className="red-flag-symptoms">
              <strong>Detected red-flag symptom{triggeredSymptoms.length > 1 ? "s" : ""}:</strong>
              <div className="red-flag-chip-row">
                {triggeredSymptoms.map((symptom) => (
                  <span key={symptom} className="red-symptom-chip">{symptom}</span>
                ))}
              </div>
            </div>
          )}

          {rf.critical_symptoms?.length > 0 && (
            <p><strong>Critical:</strong> {rf.critical_symptoms.join(", ")}</p>
          )}
          {rf.major_symptoms?.length > 0 && (
            <p><strong>Major:</strong> {rf.major_symptoms.join(", ")}</p>
          )}
          <p>{data.message}</p>
        </div>
      );
    }

    const highConf = data.top_predictions?.filter((p) => p.confidence_percent >= 70) || [];
    const mediumConf = data.top_predictions?.filter((p) => p.confidence_percent >= 50 && p.confidence_percent < 70) || [];
    const lowConf = data.top_predictions?.filter((p) => p.confidence_percent < 50) || [];
    let index = 0;

    const downloadResultsPDF = () => {
      const jsPDF = window.jspdf?.jsPDF;
      if (!jsPDF) {
        alert("PDF library not loaded. Add jsPDF CDN in index.html if you want PDF export.");
        return;
      }

      const doc = new jsPDF();
      let yPosition = 20;
      const pageHeight = doc.internal.pageSize.height;
      const margin = 15;
      const maxWidth = 180;

      doc.setFontSize(22);
      doc.setTextColor(20, 184, 166);
      doc.text("MediNLP Results", margin, yPosition);
      yPosition += 15;

      if (data.matched_symptoms?.length > 0) {
        doc.setFontSize(11);
        doc.setTextColor(71, 85, 105);
        doc.text("Symptoms Analyzed:", margin, yPosition);
        yPosition += 5;
        const symptomsText = doc.splitTextToSize(data.matched_symptoms.join(", "), maxWidth);
        doc.setFontSize(10);
        doc.setTextColor(26, 32, 44);
        doc.text(symptomsText, margin + 5, yPosition);
        yPosition += symptomsText.length * 4 + 5;
      }

      doc.setFontSize(13);
      doc.setTextColor(100, 100, 100);
      doc.text(`Analyzed ${data.matched_symptoms?.length || 0} symptoms → Found ${data.top_predictions?.length || 0} possible conditions`, margin, yPosition);
      yPosition += 10;

      const allPredictions = [...highConf, ...mediumConf, ...lowConf];
      allPredictions.forEach((pred) => {
        if (yPosition > pageHeight - 40) {
          doc.addPage();
          yPosition = 20;
        }

        const rec = pred.recommendation || {};
        const shapItems = pred.shap_explanation?.present_symptom_contributions || [];

        doc.setFontSize(15);
        doc.setTextColor(15, 23, 42);
        doc.text(`${pred.rank}. ${pred.disease}`, margin, yPosition);
        yPosition += 6;

        doc.setFontSize(12);
        doc.setTextColor(100, 100, 100);
        const confLevel = pred.confidence_percent >= 70 ? "High" : pred.confidence_percent >= 50 ? "Medium" : "Low";
        doc.text(`Confidence: ${pred.confidence_percent}% (${confLevel})`, margin + 5, yPosition);
        yPosition += 8;

        if (rec.doctor_type_patient_should_see) {
          doc.setFontSize(11);
          doc.setTextColor(71, 85, 105);
          doc.text("See:", margin + 5, yPosition);
          const seeText = doc.splitTextToSize(rec.doctor_type_patient_should_see, maxWidth - 30);
          doc.setTextColor(26, 32, 44);
          doc.text(seeText, margin + 25, yPosition);
          yPosition += seeText.length * 4 + 3;
        }

        if (rec.common_tests_to_discuss_with_clinician?.length > 0) {
          if (yPosition > pageHeight - 30) {
            doc.addPage();
            yPosition = 20;
          }
          doc.setFontSize(11);
          doc.setTextColor(71, 85, 105);
          doc.text("Tests:", margin + 5, yPosition);
          const testsText = doc.splitTextToSize(rec.common_tests_to_discuss_with_clinician.join(", "), maxWidth - 30);
          doc.setTextColor(26, 32, 44);
          doc.text(testsText, margin + 25, yPosition);
          yPosition += testsText.length * 4 + 3;
        }

        if (rec.short_care_note) {
          if (yPosition > pageHeight - 30) {
            doc.addPage();
            yPosition = 20;
          }
          doc.setFontSize(11);
          doc.setTextColor(71, 85, 105);
          doc.text("Note:", margin + 5, yPosition);
          const noteText = doc.splitTextToSize(rec.short_care_note, maxWidth - 30);
          doc.setTextColor(26, 32, 44);
          doc.text(noteText, margin + 25, yPosition);
          yPosition += noteText.length * 4 + 5;
        }

        if (shapItems.length > 0) {
          if (yPosition > pageHeight - 40) {
            doc.addPage();
            yPosition = 20;
          }

          doc.setFontSize(11);
          doc.setTextColor(71, 85, 105);
          doc.text("Symptoms Importance:", margin + 5, yPosition);
          yPosition += 5;

          const topSymptoms = [...shapItems]
            .sort((a, b) => Math.abs(parseFloat(b.contribution)) - Math.abs(parseFloat(a.contribution)))
            .slice(0, 3);

          topSymptoms.forEach((item) => {
            if (yPosition > pageHeight - 15) {
              doc.addPage();
              yPosition = 20;
            }
            const value = parseFloat(item.contribution);
            const importance = value > 0 ? "important key symptom" : "not important key symptom";
            doc.setFontSize(10);
            doc.setTextColor(26, 32, 44);
            doc.text(`• ${item.symptom}: ${importance}`, margin + 10, yPosition);
            yPosition += 4;
          });

          yPosition += 3;
        }

        doc.setDrawColor(226, 232, 240);
        doc.line(margin, yPosition, margin + maxWidth, yPosition);
        yPosition += 8;
      });

      doc.save(`MediNLP_Results_${new Date().toLocaleDateString()}.pdf`);
    };

    return (
      <div className="results-container" ref={resultsRef}>
        <div className="predictions-wrapper">
          <div className="analysis-summary">
            <p>Analyzed <strong>{data.matched_symptoms?.length || 0}</strong> symptoms → Found <strong>{data.top_predictions?.length || 0}</strong> possible conditions</p>
            {data.matched_symptoms?.length > 0 && (
              <div className="detected-symptoms">
                <span className="detected-label">Detected:</span>
                {data.matched_symptoms.map((symptom) => (
                  <span key={symptom} className="symptom-chip">{symptom}</span>
                ))}
              </div>
            )}
          </div>
          <div className="predictions-section">
            {highConf.length > 0 && (
              <>
                <div className="confidence-group-label">🟢 High Confidence</div>
                {highConf.map((pred) => <PredictionCard key={pred.rank} pred={pred} isPrimary={pred.rank === 1} index={index++} />)}
              </>
            )}
            {mediumConf.length > 0 && (
              <>
                <div className="confidence-group-label">🟡 Medium Confidence</div>
                {mediumConf.map((pred) => <PredictionCard key={pred.rank} pred={pred} index={index++} />)}
              </>
            )}
            {lowConf.length > 0 && (
              <>
                <div className="confidence-group-label">🔴 Low Confidence</div>
                {lowConf.map((pred) => <PredictionCard key={pred.rank} pred={pred} index={index++} />)}
              </>
            )}
          </div>
        </div>
        <button className="download-pdf-btn" onClick={downloadResultsPDF} title="Download entire reply as PDF">
          📥 Download Results
        </button>
      </div>
    );
  }

  if (authChecking) {
    return (
      <div className={`auth-gate ${darkMode ? "dark-mode" : ""}`}>
        <div className="auth-gate-card auth-loading-card">Loading MediNLP...</div>
      </div>
    );
  }

  if (!user && !authToken && !guestMode) {
    return (
      <AuthGate
        authMode={authMode}
        setAuthMode={setAuthMode}
        authForm={authForm}
        setAuthForm={setAuthForm}
        authError={authError}
        authLoading={authLoading}
        onSubmit={handleAuthSubmit}
        onGuest={() => setGuestMode(true)}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />
    );
  }

  return (
  <div className={`app-shell ${darkMode ? "dark-mode" : ""}`}>

    {/* Mobile sidebar open button */}
    <button
      type="button"
      className="mobile-menu-button"
      onClick={() => setMobileSidebarOpen(true)}
      aria-label="Open menu"
      title="Open menu"
    >
      ☰
    </button>

    {/* Mobile sidebar background overlay */}
    {mobileSidebarOpen && (
      <div
        className="sidebar-overlay"
        onClick={() => setMobileSidebarOpen(false)}
        aria-hidden="true"
      />
    )}

    <aside
      className={`sidebar ${mobileSidebarOpen ? "mobile-open" : ""}`}
    >
      {/* Mobile sidebar close button */}
      <button
        type="button"
        className="mobile-sidebar-close"
        onClick={() => setMobileSidebarOpen(false)}
        aria-label="Close menu"
        title="Close menu"
      >
        ×
      </button>

      <div className="sidebar-fixed-top">
          <div className="brand">
            <div className="logo">C</div>
            <div>
              <h1>CareAI</h1>
              <p>A Safety-Aware Explainable Medical Chatbot</p>
            </div>
          </div>

          <button
            className="sidebar-new-chat-btn"
            type="button"
            onClick={startNewChat}
            title="Start new chat"
          >
            <span className="new-chat-icon">✎</span>
            <span>New Chat</span>
          </button>
        </div>

        <div className="sidebar-scroll-area">
          {user ? (
            <ChatHistory
              chats={chats}
              currentChatId={currentChatId}
              onOpenChat={openSavedChat}
              onDeleteChat={deleteSavedChat}
              historyLoading={historyLoading}
            />
          ) : (
            <div className="guest-mode-card">
              <strong>Guest Mode</strong>
              <p>You can chat without login. To restore previous chats later, please login or register.</p>
            </div>
          )}

          <div className="info-section">
            <h3>How to use:</h3>
            <ul className="tips">
              <li>Write naturally: English, Banglish, or Bangla</li>
              <li>Example: amar jor ase, kashi hocche</li>
              <li>Press Enter or click Send</li>
            </ul>
          </div>

          <div className="notice">
            <strong>⚠️ Disclaimer</strong>
            <p>This is not a medical diagnosis. Consult healthcare professionals for actual medical advice.</p>
          </div>
        </div>

        <div className="sidebar-fixed-bottom">
          {user ? (
            <AuthPanel
              user={user}
              authMode={authMode}
              setAuthMode={setAuthMode}
              authForm={authForm}
              setAuthForm={setAuthForm}
              authError={authError}
              onSubmit={handleAuthSubmit}
              onLogout={logout}
            />
          ) : (
            <button className="auth-secondary-btn guest-login-btn" onClick={() => setGuestMode(false)}>Login / Register</button>
          )}

          <button
            className="dark-mode-toggle"
            onClick={() => setDarkMode(!darkMode)}
            title={`Switch to ${darkMode ? "light" : "dark"} mode`}
            aria-label={`Toggle ${darkMode ? "light" : "dark"} mode`}
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <div className="chat-title-row">
            <div>
              <h2>Medical Symptom Assistant</h2>
              <p>Describe your symptoms for AI-powered health insights</p>
            </div>
            <div className="header-status">
              {user ? (
                <>
                  <strong>{autoSaveStatus || "Logged in"}</strong>
                  <span>History restore enabled</span>
                </>
              ) : (
                <>
                  <strong>Guest Mode</strong>
                  <span>Login to restore chats</span>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="messages">
          {messages.map((m) => (
            <ChatBubble key={m.id} role={m.role} timestamp={m.timestamp}>
              {m.type === "result" ? (
                renderBotResult(m.data)
              ) : (
                <div className="text-message">
                  {String(m.content || "").split("\n").map((line, i) => <div key={i}>{line}</div>)}
                </div>
              )}
            </ChatBubble>
          ))}
          {loading && messages[messages.length - 1]?.role === "user" && (
            <TypingIndicator />
            )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <div className="input-wrapper">
            <input
              value={input}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              placeholder="Example: amar jor ase, kashi hocche / chest pain..."
              disabled={loading}
              aria-label="Type your symptoms"
            />
            {showSuggestions && suggestions.length > 0 && (
              <div className="autocomplete-dropdown">
                {suggestions.map((s) => (
                  <div key={s} className="suggestion-item" onClick={() => addSuggestion(s)}>
                    {s}
                  </div>
                ))}
              </div>
            )}
            <button
              type="button"
              className={`voice-icon-btn ${isListening ? "listening" : ""}`}
              onClick={startVoiceInput}
              disabled={loading}
              title={isListening ? "Stop listening" : "Start voice input"}
              aria-label="Toggle voice input"
            >
              <MicIcon listening={isListening} />
            </button>
          </div>

          <button onClick={sendMessage} disabled={loading} className={`send-btn ${loading ? "loading" : ""}`} aria-label="Send message">
            {loading ? "⏳" : "➤"}
          </button>

          <div className="composer-meta">
            <select
              className="voice-language-select"
              value={voiceLanguage}
              onChange={(e) => setVoiceLanguage(e.target.value)}
              disabled={isListening}
              title="Voice recognition language"
            >
              <option value="bn-BD">বাংলা voice</option>
              <option value="en-US">English / Banglish voice</option>
            </select>
            {isListening && <span className="voice-status">Listening...</span>}
            {voiceError && <span className="voice-error">{voiceError}</span>}
            {!voiceError && !isListening && <span className="voice-hint">Speak one sentence clearly, then wait for it to appear.</span>}
          </div>
        </div>
      </main>
    </div>
  );
}
