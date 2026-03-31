import React, { useState } from 'react';
import { motion } from 'framer-motion';
import GoogleAuth from './auth/GoogleAuth';
import ChartPage from './pages/ChartPage';

export default function App() {
  const [user, setUser] = useState(null);

  // ── Unauthenticated: Sign-in page ──────────────────────────────────────
  if (!user) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        {/* Header */}
        <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-4 bg-bg/75 backdrop-blur-xl border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse shadow-[0_0_8px_rgba(0,210,106,0.4)]" />
            <span className="text-lg font-semibold tracking-tight text-text-primary">NIFTY ML Research</span>
          </div>
        </header>

        {/* Sign-in card */}
        <div className="flex-1 flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="bg-surface/65 backdrop-blur-2xl border border-border rounded-2xl p-12 max-w-md w-full mx-4 text-center space-y-6 shadow-[0_8px_32px_rgba(0,0,0,0.45)]"
          >
            <h1 className="text-3xl font-semibold tracking-tight text-text-primary">Welcome back</h1>
            <p className="text-sm text-text-secondary leading-relaxed max-w-xs mx-auto">
              Sign in with your Google account to access the NIFTY chart viewer.
            </p>
            <div className="w-10 h-px bg-border mx-auto" />
            <div className="flex justify-center">
              <GoogleAuth user={user} onLogin={setUser} onLogout={() => setUser(null)} />
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  // ── Authenticated: Chart only ───────────────────────────────────────────
  return (
    <div className="min-h-screen bg-bg flex flex-col">
      {/* Header with user info */}
      <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-3 bg-bg/75 backdrop-blur-xl border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent animate-pulse shadow-[0_0_8px_rgba(0,210,106,0.4)]" />
          <span className="text-lg font-semibold tracking-tight text-text-primary">NIFTY ML Research</span>
        </div>
        <div className="flex items-center gap-3">
          <GoogleAuth user={user} onLogin={setUser} onLogout={() => setUser(null)} />
        </div>
      </header>

      {/* Chart page — the only authenticated content */}
      <ChartPage />
    </div>
  );
}
