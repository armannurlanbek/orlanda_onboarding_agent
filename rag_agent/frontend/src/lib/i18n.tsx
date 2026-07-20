/**
 * Minimal i18n for the client cabinet: English + Hebrew (RTL).
 * The provider stamps lang/dir on <html>, so the whole layout mirrors in Hebrew.
 * Employee-facing pages stay Russian and do not use this provider's strings.
 */
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Lang = "en" | "he";

const LANG_KEY = "orlanda.lang";

const dict: Record<string, { en: string; he: string }> = {
  "brand.cabinet": { en: "Client Cabinet", he: "פורטל לקוחות" },
  "nav.tasks": { en: "Tasks", he: "משימות" },
  "nav.assistant": { en: "Assistant", he: "עוזר" },
  "nav.progress": { en: "Progress", he: "התקדמות" },
  "nav.feedback": { en: "Feedback", he: "משוב" },
  "nav.settings": { en: "Settings", he: "הגדרות" },
  "common.logout": { en: "Log out", he: "התנתקות" },
  "common.loading": { en: "Loading…", he: "טוען…" },
  "common.error": { en: "Something went wrong. Please try again.", he: "משהו השתבש. נסו שוב." },
  "common.project": { en: "Project", he: "פרויקט" },
  "common.allProjects": { en: "All projects", he: "כל הפרויקטים" },

  "tasks.title": { en: "Project tasks", he: "משימות הפרויקט" },
  "tasks.updated": { en: "Updated", he: "עודכן" },
  "tasks.empty": { en: "No tasks to show yet.", he: "אין משימות להצגה עדיין." },
  "tasks.loadError": { en: "Could not load tasks for this project.", he: "לא ניתן לטעון משימות לפרויקט זה." },
  "tasks.search": { en: "Search in all columns…", he: "חיפוש בכל העמודות…" },
  "tasks.addFilter": { en: "Filter", he: "סינון" },
  "tasks.pickValue": { en: "Pick a value…", he: "בחרו ערך…" },
  "tasks.clearFilters": { en: "Clear all", he: "נקה הכל" },
  "tasks.noMatches": { en: "No tasks match your search.", he: "אין משימות התואמות לחיפוש." },
  "tasks.selectAll": { en: "Select all", he: "בחר הכל" },
  "tasks.clearColumn": { en: "Clear", he: "נקה" },
  "tasks.filterSearch": { en: "Search values…", he: "חיפוש ערכים…" },

  "dash.completion": { en: "Completion", he: "התקדמות" },
  "dash.tasksDone": { en: "tasks done", he: "משימות הושלמו" },
  "dash.byArea": { en: "by m²", he: "לפי מ״ר" },
  "dash.cladding": { en: "Cladding", he: "חיפוי" },
  "dash.dispatched": { en: "dispatched", he: "נשלח" },
  "dash.dispatches": { en: "Dispatches · 6 weeks", he: "משלוחים · 6 שבועות" },
  "dash.thisWeek": { en: "this week", he: "השבוע" },
  "dash.waitingOnYou": { en: "Waiting on you", he: "ממתין לכם" },
  "dash.updated": { en: "updated", he: "עודכן" },

  "chips.all": { en: "All", he: "הכל" },
  "chips.inProgress": { en: "In progress", he: "בתהליך" },
  "chips.waitingOnYou": { en: "Waiting on you", he: "ממתין לכם" },
  "chips.dispatched7d": { en: "Dispatched 7d", he: "נשלח 7 ימים" },

  "suggest.whatsNew": { en: "What's new?", he: "מה חדש?" },
  "suggest.status": { en: "Project status", he: "סטטוס הפרויקט" },
  "suggest.drawings": { en: "Latest drawings", he: "שרטוטים אחרונים" },

  "card.total": { en: "Cladding total", he: "סה״כ חיפוי" },
  "card.dispatched": { en: "Dispatched", he: "נשלח" },
  "card.nextDeadline": { en: "Next deadline", he: "מועד הבא" },
  "card.lastDispatch": { en: "Last dispatch", he: "משלוח אחרון" },
  "card.progressPage": { en: "Progress page", he: "דף התקדמות" },
  "card.openTasks": { en: "Open tasks", he: "משימות פתוחות" },

  "assistant.title": { en: "Project assistant", he: "עוזר הפרויקט" },
  "assistant.placeholder": { en: "Ask about your projects…", he: "שאלו על הפרויקטים שלכם…" },
  "assistant.send": { en: "Send", he: "שלח" },
  "assistant.thinking": { en: "Thinking…", he: "חושב…" },
  "assistant.newChat": { en: "New conversation", he: "שיחה חדשה" },
  "assistant.deleteChat": { en: "Delete conversation", he: "מחיקת שיחה" },
  "assistant.noChats": { en: "No conversations yet", he: "אין שיחות עדיין" },
  "assistant.hello": {
    en: "Hi! Ask me anything about your projects: tasks, statuses, drawings, files, documents.",
    he: "שלום! שאלו אותי כל דבר על הפרויקטים שלכם: משימות, סטטוסים, שרטוטים, קבצים, מסמכים.",
  },

  "progress.title": { en: "Project progress", he: "התקדמות הפרויקט" },
  "progress.empty": { en: "No progress page for this project yet.", he: "אין עדיין דף התקדמות לפרויקט זה." },
  "progress.openTab": { en: "Open in a new tab", he: "פתיחה בכרטיסייה חדשה" },

  "feedback.title": { en: "Feedback", he: "משוב" },
  "feedback.text": {
    en: "We'd love to hear from you — questions, ideas or concerns about your project.",
    he: "נשמח לשמוע מכם — שאלות, רעיונות או הערות על הפרויקט שלכם.",
  },
  "feedback.emailUs": { en: "Email us", he: "כתבו לנו" },
  "feedback.openForm": { en: "Open feedback form", he: "פתיחת טופס משוב" },
  "feedback.formEn": { en: "Feedback form (English)", he: "טופס משוב (אנגלית)" },
  "feedback.formHe": { en: "Feedback form (Hebrew)", he: "טופס משוב (עברית)" },

  "register.title": { en: "Create your account", he: "יצירת חשבון" },
  "register.subtitle": { en: "You were invited to the Orlanda client portal", he: "הוזמנתם לפורטל הלקוחות של Orlanda" },
  "register.projects": { en: "Your projects", he: "הפרויקטים שלכם" },
  "register.email": { en: "E-mail", he: "אימייל" },
  "register.password": { en: "Password", he: "סיסמה" },
  "register.repeat": { en: "Repeat password", he: "אימות סיסמה" },
  "register.submit": { en: "Create account", he: "צור חשבון" },
  "register.mismatch": { en: "Passwords do not match", he: "הסיסמאות אינן תואמות" },
  "register.passwordHint": {
    en: "At least 8 characters, with letters and digits",
    he: "לפחות 8 תווים, כולל אותיות וספרות",
  },
  "register.invalidInvite": { en: "This invite link is not valid.", he: "קישור ההזמנה אינו תקף." },
  "register.checking": { en: "Checking your invite…", he: "בודק את ההזמנה…" },
  "register.haveAccount": { en: "Already have an account?", he: "כבר יש לכם חשבון?" },
  "register.signIn": { en: "Sign in", he: "התחברות" },

  "settings.title": { en: "Settings", he: "הגדרות" },
  "settings.account": { en: "Account", he: "חשבון" },
  "settings.email": { en: "E-mail", he: "אימייל" },
  "settings.newEmail": { en: "New e-mail", he: "אימייל חדש" },
  "settings.currentPassword": { en: "Current password", he: "סיסמה נוכחית" },
  "settings.changeEmail": { en: "Change e-mail", he: "שינוי אימייל" },
  "settings.password": { en: "Password", he: "סיסמה" },
  "settings.newPassword": { en: "New password", he: "סיסמה חדשה" },
  "settings.repeatPassword": { en: "Repeat new password", he: "אימות סיסמה חדשה" },
  "settings.changePassword": { en: "Change password", he: "שינוי סיסמה" },
  "settings.saved": { en: "Saved", he: "נשמר בהצלחה" },
};

type I18nState = {
  lang: Lang;
  dir: "ltr" | "rtl";
  setLang: (l: Lang) => void;
  t: (key: string) => string;
};

const I18nContext = createContext<I18nState | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    const stored = localStorage.getItem(LANG_KEY);
    return stored === "he" ? "he" : "en";
  });
  const dir: "ltr" | "rtl" = lang === "he" ? "rtl" : "ltr";

  useEffect(() => {
    document.documentElement.setAttribute("lang", lang);
    document.documentElement.setAttribute("dir", dir);
    return () => {
      // Employee pages are LTR Russian — restore defaults when leaving the cabinet.
      document.documentElement.setAttribute("lang", "ru");
      document.documentElement.setAttribute("dir", "ltr");
    };
  }, [lang, dir]);

  const setLang = (l: Lang) => {
    localStorage.setItem(LANG_KEY, l);
    setLangState(l);
  };

  const t = (key: string) => dict[key]?.[lang] ?? dict[key]?.en ?? key;

  return <I18nContext.Provider value={{ lang, dir, setLang, t }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used inside I18nProvider");
  return ctx;
}
