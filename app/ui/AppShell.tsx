'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { supabase } from '../lib/supabaseClient';
import { Smartphone, LogOut, Bookmark, MessageSquare, Inbox, UserCircle, Bot, Menu, X } from 'lucide-react';

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<any>(null);
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  const pathname = usePathname();

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

  // Close mobile menu on route change
  React.useEffect(() => {
    setIsMenuOpen(false);
  }, [pathname]);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.href = '/';
  };

  const navLinks = [
    { href: '/marketplace', label: 'Marketplace' },
    { href: '/add', label: 'Sell' },
    { href: '/recommendation', label: 'Recommendations' },
    { href: '/about', label: 'About' },
    { href: '/contactus', label: 'Contact Us' },
  ];

  const isActive = (href: string) => pathname === href;

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header
        className="glass-panel sticky top-0 z-50 border-b border-gray-800"
        style={{
          background: 'rgba(14, 14, 16, 0.85)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between gap-4">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 text-xl font-bold shrink-0">
            <Smartphone className="text-[#f7f435]" />
            <span className="text-[#f7f435]">IntelliFone</span>
          </Link>

          {/* Desktop Nav Links */}
          <nav className="hidden md:flex gap-1">
            {navLinks.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  isActive(href)
                    ? 'text-[#f7f435] bg-[#f7f435]/10'
                    : 'text-gray-300 hover:text-white hover:bg-white/5'
                }`}
              >
                {label}
              </Link>
            ))}
          </nav>

          {/* Desktop Actions */}
          <div className="hidden md:flex gap-2 items-center">
            {user ? (
              <>
                {/* AI Chat */}
                <Link
                  href="/chat"
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    isActive('/chat')
                      ? 'bg-[#f7f435] text-black'
                      : 'text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10'
                  }`}
                >
                  <Bot className="w-4 h-4" />
                  <span>AI Chat</span>
                </Link>

                {/* Inbox */}
                <Link
                  href="/chats"
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    isActive('/chats')
                      ? 'bg-[#f7f435] text-black'
                      : 'text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10'
                  }`}
                >
                  <Inbox className="w-4 h-4" />
                  <span>Inbox</span>
                </Link>

                {/* Saved */}
                <Link
                  href="/saved"
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    isActive('/saved')
                      ? 'bg-[#f7f435] text-black'
                      : 'text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10'
                  }`}
                >
                  <Bookmark className="w-4 h-4" />
                  <span>Saved</span>
                </Link>

                {/* Profile */}
                <Link
                  href="/profile"
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    isActive('/profile')
                      ? 'bg-[#f7f435] text-black'
                      : 'text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10'
                  }`}
                >
                  <UserCircle className="w-4 h-4" />
                  <span>Profile</span>
                </Link>

                {/* Logout */}
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-gray-400 hover:text-red-400 hover:bg-red-400/10 transition-all"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </>
            ) : (
              <>
                <Link
                  href="/signin"
                  className="text-sm font-medium text-gray-300 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-all"
                >
                  Login
                </Link>
                <Link
                  href="/signup"
                  className="px-4 py-2 rounded-lg text-sm font-semibold text-black"
                  style={{ backgroundColor: '#f7f435' }}
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>

          {/* Mobile toggle */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden p-2 rounded-lg text-gray-300 hover:bg-white/10 transition-all"
            aria-label="Toggle menu"
          >
            {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden border-t border-gray-800 bg-[#0e0e10]">
            <div className="px-4 py-3 flex flex-col gap-1">
              {navLinks.map(({ href, label }) => (
                <Link
                  key={href}
                  href={href}
                  className={`px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive(href) ? 'text-[#f7f435] bg-[#f7f435]/10' : 'text-gray-300 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {label}
                </Link>
              ))}

              <div className="my-2 border-t border-gray-800" />

              {user ? (
                <>
                  <Link href="/chat" className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10 transition-all">
                    <Bot className="w-4 h-4" /> AI Chat
                  </Link>
                  <Link href="/chats" className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10 transition-all">
                    <Inbox className="w-4 h-4" /> Inbox
                  </Link>
                  <Link href="/saved" className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10 transition-all">
                    <Bookmark className="w-4 h-4" /> Saved
                  </Link>
                  <Link href="/profile" className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-300 hover:text-[#f7f435] hover:bg-[#f7f435]/10 transition-all">
                    <UserCircle className="w-4 h-4" /> Profile
                  </Link>
                  <button onClick={handleLogout} className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-400 hover:text-red-400 hover:bg-red-400/10 transition-all w-full text-left">
                    <LogOut className="w-4 h-4" /> Logout
                  </button>
                </>
              ) : (
                <>
                  <Link href="/signin" className="px-3 py-2.5 rounded-lg text-sm font-medium text-gray-300 hover:text-white hover:bg-white/5 transition-all">Login</Link>
                  <Link href="/signup" className="px-3 py-2.5 rounded-lg text-sm font-semibold text-black text-center" style={{ backgroundColor: '#f7f435' }}>Sign Up</Link>
                </>
              )}
            </div>
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
