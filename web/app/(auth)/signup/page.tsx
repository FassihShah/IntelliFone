// 'use client';
// import { useState } from 'react';
// import { supabase } from '../../lib/supabaseClient';
// import { useRouter } from 'next/navigation';

// export default function SignupPage() {
//   const [fullName, setFullName] = useState('');
//   const [email, setEmail] = useState('');
//   const [password, setPassword] = useState('');
//   const router = useRouter();

//   const handleSignup = async (e: React.FormEvent) => {
//     e.preventDefault();
//     // 1) Sign up with email + password (Supabase handles hashing)
//     const { data, error } = await supabase.auth.signUp({
//       email,
//       password,
//       options: { data: { fullName } } // optional metadata
//     });

//     if (error) {
//       alert(error.message);
//       return;
//     }

//     // 2) Create profile row in `profiles` table (client-side)
//     // Wait for user id from session
//     const {
//       data: { user },
//     } = await supabase.auth.getUser();

//     if (user) {
//                 const { error: upsertError } = await supabase
//             .from('profiles')
//             .upsert({
//                 id: user.id,
//                 full_name: fullName,
//                 created_at: new Date().toISOString()
//             }); // ✅ just remove returning


//       if (upsertError) {
//         console.error('Profile create error', upsertError);
//       }
//     }

//     // Optional: redirect to check email page or dashboard
//     router.push('/auth/check-email');
//   };

//   return (
//     <form onSubmit={handleSignup}>
//       <input placeholder="Full name" value={fullName} onChange={e=>setFullName(e.target.value)} required/>
//       <input placeholder="Email" value={email} onChange={e=>setEmail(e.target.value)} required/>
//       <input placeholder="Password" type="password" value={password} onChange={e=>setPassword(e.target.value)} required/>
//       <button type="submit">Sign up</button>
//     </form>
//   );
// }
'use client';
import GoogleButton from "../../components/auth/GoogleButton";
import { useState } from 'react';
import { supabase } from '../../lib/supabaseClient';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function SignupPage() {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { fullName } }
    });

    if (error) {
      alert(error.message);
      return;
    }

    const { data: { user } } = await supabase.auth.getUser();

    if (user) {
      await supabase.from('profiles').upsert({
        id: user.id,
        full_name: fullName,
        created_at: new Date().toISOString()
      });
    }

    router.push('/auth/check-email');
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-gray-900/70 border border-[#f7f435]/30 rounded-2xl p-8 shadow-xl">

        <div className="text-center mb-8">
          <div className="w-14 h-14 bg-[#f7f435] rounded-2xl flex items-center justify-center mx-auto text-black text-2xl font-bold">IF</div>
          <h1 className="text-3xl font-bold text-white mt-4">Create Account</h1>
          <p className="text-gray-400 text-sm mt-1">Join IntelliFone Marketplace</p>
        </div>

        <form onSubmit={handleSignup} className="space-y-4">

          <div>
            <label className="text-gray-300 text-sm">Full Name</label>
            <input
              className="w-full mt-1 px-4 py-3 bg-black border border-gray-800 rounded-xl text-white focus:border-[#f7f435] outline-none"
              placeholder="John Doe"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
            />
          </div>

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

          <button
            type="submit"
            className="w-full bg-[#f7f435] text-black font-bold py-3 rounded-xl mt-4 text-lg hover:opacity-90 transition-all neon-glow"
          >
            Sign Up
          </button>
        </form>
        <div className="mt-4">
                <GoogleButton />
        </div>


        <p className="text-gray-400 mt-6 text-center text-sm">
          Already have an account?{' '}
          <Link href="/auth/login" className="text-[#f7f435] hover:underline">
            Login
          </Link>
        </p>
      </div>
    </div>
  );
}
