"use client";

import { supabase } from '../../lib/supabaseClient';

export default function GoogleButton() {
  const handleGoogleLogin = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/auth/callback`
      }
    });

    if (error) {
      alert(error.message);
    }
  };

  return (
    <button
      onClick={handleGoogleLogin}
      className="w-full bg-white text-black font-semibold py-3 rounded-xl flex items-center justify-center gap-2 border border-gray-300 hover:bg-gray-100 transition-all"
    >
      <img src="/google.svg" className="w-5 h-5" alt="Google" />
      Sign up with Google
    </button>
  );
}
