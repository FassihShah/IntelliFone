'use client';

import React from 'react';
import Link from 'next/link';
import { supabase } from '../lib/supabaseClient';
import { Smartphone, LogOut,Bookmark, MessageSquare,Inbox,UserCircle} from 'lucide-react';

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<any>(null);
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  React.useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      setUser(data?.user ?? null);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => {
      setUser(session?.user ?? null);
    });

    return () => {
      sub?.subscription.unsubscribe();
    };
  }, []);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.href = '/';
  };

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="glass-panel sticky top-0 z-50 border-b border-gray-800"
      style={{
        background: "rgba(26, 26, 26, 0.8)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
        }}
      >
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 text-xl font-bold">
            <Smartphone className="text-[#f7f435]" />
            <span className="text-[#f7f435]">IntelliFone</span>
          </Link>

          <nav className="hidden md:flex gap-6">
            <Link href="/marketplace">Marketplace</Link>
            <Link href="/add">Sell</Link>
            <Link href="/recommendation">Recommendations</Link>
            <Link href="/about">About</Link>
            <Link href="/contactus">Contact Us</Link>
          </nav>

          <div className="hidden md:flex gap-3 items-center">
            {user ? (
              <>
                <Link href="/chats" className="font-semibold">
                  <Inbox className="w-4 h-4 inline-block mr-1" /> Inbox
                </Link>
                <Link href="/saved"
                className='font-semibold'
                ><Bookmark className="w-4 h-4 inline-block mr-1" /> Saved</Link>
                <Link href="/profile" className="font-semibold"> 
                   <UserCircle className="w-4 h-4 inline-block mr-1" />Profile
                </Link>
                {/* <span className="flex items-center gap-2">
                  <User className="w-4 h-4" />
                  {user.email}
                </span> */}
                

                
                
                <button onClick={handleLogout} className="hover:text-red-400">
                  <LogOut />
                </button>
              </>
            ) : (
              <>
                <Link href="/signin">Login</Link>
                <Link href="/signup" className="yellow-btn px-4 py-2 rounded">
                  Sign Up
                </Link>
              </>
            )}
          </div>

          {/* Mobile */}
          <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="md:hidden">
            ☰
          </button>
        </div>

        {isMenuOpen && (
          <div className="md:hidden px-4 py-4 border-t border-gray-800">
            <Link href="/" className="block">Home</Link>
            <Link href="/marketplace" className="block">Marketplace</Link>
            <Link href="/sell" className="block">Sell</Link>
            {user ? (
              <button onClick={handleLogout} className="block mt-2">
                Logout
              </button>
            ) : (
              <Link href="/signin">Login</Link>
            )}
          </div>
        )}
      </header>

      <main>{children}</main>

      {/* Footer */}
      <footer className="glass-panel border-t border-gray-800 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">

            {/* Brand */}
            <div>
              <div className="flex items-center gap-2 text-xl font-bold mb-4">
                <Smartphone className="w-6 h-6 text-[#f7f435]" />
                <span className="text-[#f7f435]">IntelliFone</span>
              </div>
              <p className="text-gray-400 text-sm">
                Buy & sell used phones with confidence using AI-powered verification.
              </p>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className="font-semibold mb-4">Quick Links</h3>
              <div className="flex flex-col gap-2 text-sm text-gray-400">
                <Link href="/">Home</Link>
                <Link href="/marketplace">Marketplace</Link>
                <Link href="/add">Sell Phone</Link>
                <Link href="/about">About Us</Link>
               
              </div>
            </div>

            {/* Support */}
            <div>
              <h3 className="font-semibold mb-4">Support</h3>
              <div className="flex flex-col gap-2 text-sm text-gray-400">
                <Link href="/contactus">Contact Us</Link>
                <Link href="/helpcenter">Help Center</Link>
                <Link href="/privacypolicy">Privacy Policy</Link>
                <Link href="/termsofservice">Terms of Service</Link>
              </div>
            </div>

            {/* Social */}
            <div>
              <h3 className="font-semibold mb-4">Connect</h3>
              <div className="flex flex-col gap-2 text-sm text-gray-400">
                <a href="#" className="hover:text-[#f7f435]">Twitter</a>
                <a href="#" className="hover:text-[#f7f435]">Facebook</a>
                <a href="#" className="hover:text-[#f7f435]">Instagram</a>
                <a href="#" className="hover:text-[#f7f435]">LinkedIn</a>
              </div>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
            © 2025 IntelliFone. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
