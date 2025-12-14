'use client';
import { useEffect } from 'react';
import { supabase } from '../../lib/supabaseClient';
import { useRouter } from 'next/navigation';

export default function AuthCallback() {
  const router = useRouter();

  useEffect(() => {
    const getUser = async () => {
      const { data } = await supabase.auth.getSession();
      const user = data.session?.user;

      if (user) {
        await supabase.from('profiles').upsert({
          id: user.id,
          full_name: user.user_metadata?.full_name || user.user_metadata?.name,
          avatar_url: user.user_metadata?.avatar_url || user.user_metadata?.picture,
          created_at: new Date().toISOString(),
        }, { onConflict: 'id' });

        router.push('/dashboard'); // redirect to dashboard
      } else {
        router.push('/signup'); // fallback
      }
    };

    getUser();
  }, [router]);

  return <p>Signing you in...</p>;
}
