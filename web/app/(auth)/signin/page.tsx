'use client';
import GoogleButton from "../../components/auth/GoogleButton";
import { useState } from 'react';
import { supabase } from '../../lib/supabaseClient';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function SigninPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSignin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password
    });

    if (error) {
      setError(error.message);
      return;
    }

    // If sign-in returned a session or user, redirect to home.
    if (data?.session || data?.user) {
      router.push('/home');
      return;
    }

    // No session returned (e.g. magic link flow) — prompt the user.
    setError('Check your email to complete sign-in.');
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-gray-900/70 border border-[#f7f435]/30 rounded-2xl p-8 shadow-xl">
        <div className="text-center mb-8">
          <div className="w-14 h-14 bg-[#f7f435] rounded-2xl flex items-center justify-center mx-auto text-black text-2xl font-bold">IF</div>
          <h1 className="text-3xl font-bold text-white mt-4">Sign In</h1>
          <p className="text-gray-400 text-sm mt-1">Welcome back to IntelliFone Marketplace</p>
        </div>
        <form onSubmit={handleSignin} className="space-y-4">
          <div>
            <label className="text-gray-300 text-sm">Email Address</label>
            <input
              className="w-full mt-1 px-4 py-3 bg-black border border-gray-800 rounded-xl text-white focus:border-[#f7f435] outline-none"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="text-gray-300 text-sm">Password</label>
            <input
              type="password"
              className="w-full mt-1 px-4 py-3 bg-black border border-gray-800 rounded-xl text-white focus:border-[#f7f435] outline-none"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && <div className="text-red-500 text-sm">{error}</div>}
          <button
            type="submit"
            className="w-full bg-[#f7f435] text-black font-bold py-3 rounded-xl mt-4 text-lg hover:opacity-90 transition-all neon-glow"
          >
            Sign In
          </button>
        </form>
        <div className="mt-4">
          <GoogleButton />
        </div>
        <p className="text-gray-400 mt-6 text-center text-sm">
          Don't have an account?{' '}
          <Link href="/signup" className="text-[#f7f435] hover:underline">
            Sign Up
          </Link>
        </p>
      </div>
    </div>
  );
}
