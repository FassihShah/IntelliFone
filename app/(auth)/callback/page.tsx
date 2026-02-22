'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function AuthCallback() {
  const router = useRouter();

  useEffect(() => {
    // Show success for 3 seconds then redirect to home
    const timer = setTimeout(() => {
      router.push('/');
    }, 3000);
   
    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div style={{
      background: '#0B0B0B',
      color: '#fff',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      fontFamily: 'Arial, sans-serif',
      textAlign: 'center',
      padding: '20px'
    }}>
      <div style={{
        width: '80px',
        height: '80px',
        borderRadius: '50%',
        background: '#cac71f20',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: '24px',
        fontSize: '40px'
      }}>
        ✓
      </div>
      <h1 style={{ color: '#cac71f', marginBottom: '12px', fontSize: '28px' }}>
        Authentication Successful
      </h1>
      <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '16px', maxWidth: '400px' }}>
        You can close this window and return to the IntelliFone app.
      </p>
    </div>
  );
}