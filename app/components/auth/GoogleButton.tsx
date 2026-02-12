"use client";

import { supabase } from '../../lib/supabaseClient';
import { FcGoogle } from "react-icons/fc";

export default function GoogleButton() {
  const handleGoogleLogin = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/`
      }
    });

    if (error) {
      alert(error.message);
    }
  };

  return (
    <button
      onClick={handleGoogleLogin}
      className="w-full flex items-center gap-2 bg-white text-black font-semibold py-3 rounded-xl flex items-center justify-center gap-2 border border-gray-300 hover:bg-gray-100 transition-all"
    >
      <FcGoogle className="w-5 h-5" />
      Sign Up / Sign In with Google
    </button>

      
  );
}
