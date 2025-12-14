"use client"
import React from 'react';
import ProductCard from '../components/card/ProductCard';
import { supabase } from '../lib/supabaseClient';
import Link from 'next/link';
import SearchBar from '../components/SearchBar';

export default function Home() {
  const [searchQuery, setSearchQuery] = React.useState('');
  const [carouselIndex, setCarouselIndex] = React.useState(0);

  const [phones, setPhones] = React.useState<any[]>([]);
  const [user, setUser] = React.useState<any>(null);

  React.useEffect(() => {
    // Placeholder: replace with real data fetching logic.
    setPhones([]);

    let mounted = true;
    supabase.auth.getUser().then(({ data }) => {
      if (mounted) setUser(data?.user ?? null);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => {
      mounted = false;
      if (sub?.subscription) sub.subscription.unsubscribe();
    };
  }, []);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    // Navigate to marketplace with query
    window.location.href = '/marketplace?search=' + encodeURIComponent(query);
  };

  const nextSlide = () => {
    setCarouselIndex((prev) => (prev + 1) % Math.max(1, phones.length - 2));
  };

  const prevSlide = () => {
    setCarouselIndex((prev) => (prev - 1 + Math.max(1, phones.length - 2)) % Math.max(1, phones.length - 2));
  };

  const visiblePhones = phones.slice(carouselIndex, carouselIndex + 3);

  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav className="w-full bg-black/60 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link href="/" className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#f7f435] rounded-xl flex items-center justify-center text-black font-bold">IF</div>
                <span className="font-bold text-white">IntelliFone</span>
              </Link>
            </div>

            <div className="hidden md:flex items-center gap-6">
              <Link href="/marketplace" className="text-gray-300 hover:text-white">Marketplace</Link>
              <Link href="/ai-verification" className="text-gray-300 hover:text-white">AI Verification</Link>
              <Link href="/about" className="text-gray-300 hover:text-white">About</Link>
            </div>

            <div className="flex items-center gap-3">
              {user ? (
                <button
                  onClick={async () => {
                    await supabase.auth.signOut();
                    setUser(null);
                    window.location.href = '/home';
                  }}
                  className="px-3 py-1 rounded-lg bg-[#f7f435] text-black font-semibold"
                >
                  Logout
                </button>
              ) : (
                <>
                  <Link href="/signin" className="text-gray-300 hover:text-white">Sign In</Link>
                  <Link href="/signup" className="px-3 py-1 rounded-lg bg-[#f7f435] text-black font-semibold">Sign Up</Link>
                </>
              )}
            </div>
          </div>
        </div>
      </nav>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 relative z-10">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
                Buy & Sell Used Phones <span className="text-[#f7f435]">with Confidence</span>
              </h1>
              <p className="text-xl text-gray-400 mb-8">
                AI-powered verification ensures every phone is checked, graded, and guaranteed. Trade smarter with IntelliFone.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link 
                  href="/marketplace"
                  className="px-8 py-4 rounded-xl  text-lg neon-glow text-black font-semibold flex items-center justify-center gap-2 disabled:opacity-50 transition"
                   style={{
                        backgroundColor: "#f7f434",
                  }}
                >
                  Get Started
                </Link>
                <Link 
                  href="/ai-verification"
                  className="px-8 py-4 rounded-xl glass-panel border border-[#f7f435] font-semibold text-lg "
                >
                  Verify My Phone
                </Link>
              </div>
            </div>
            
            <div className="relative flex items-center justify-center p-8">
              <div className="absolute inset-0 bg-gradient-to-br from-[#f7f435]/20 via-black to-black rounded-3xl blur-3xl"></div>
              
              <div className="relative">
                {/* Phone Frame */}
                <div className="relative w-[320px] h-[640px] bg-gray-900 rounded-[3rem] border-8 border-gray-800 shadow-2xl overflow-hidden">
                  {/* Notch */}
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 w-40 h-7 bg-black rounded-b-3xl z-10"></div>
                  
                  {/* Screen Content */}
                  <div className="w-full h-full bg-black p-6 pt-12 overflow-y-auto">
                    {/* App Header */}
                    <div className="flex items-center justify-between mb-8">
                      <div className="flex items-center gap-2">
                        <div className="w-10 h-10 bg-[#f7f435] rounded-xl flex items-center justify-center text-black font-bold">IF</div>
                        <span className="font-bold text-white text-lg">IntelliFone</span>
                      </div>
                      <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center">
                        <div className="w-2 h-2 bg-[#f7f435] rounded-full"></div>
                      </div>
                    </div>

                    {/* Featured Phone Card */}
                    <div className="bg-gray-900 rounded-2xl p-4 mb-6 border border-gray-800">
                      <div className="w-full h-32 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl mb-3 flex items-center justify-center">
                        <div className="text-4xl">📱</div>
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white font-semibold text-sm">iPhone 13 Pro</span>
                        <div className="flex items-center gap-1 bg-[#f7f435]/20 px-2 py-1 rounded-full">
                          <div className="w-2 h-2 bg-[#f7f435] rounded-full"></div>
                          <span className="text-[#f7f435] text-xs font-semibold">Verified</span>
                        </div>
                      </div>
                      <div className="text-[#f7f435] font-bold text-xl mb-3">Rs. 150,000</div>
                      <div className="flex gap-2 mb-3">
                        <div className="flex-1 bg-black rounded-lg p-2 text-center">
                          <div className="text-[#f7f435] text-xs font-semibold">18/20</div>
                          <div className="text-gray-500 text-xs">Score</div>
                        </div>
                        <div className="flex-1 bg-black rounded-lg p-2 text-center">
                          <div className="text-[#f7f435] text-xs font-semibold">95%</div>
                          <div className="text-gray-500 text-xs">Battery</div>
                        </div>
                        <div className="flex-1 bg-black rounded-lg p-2 text-center">
                          <div className="text-green-500 text-xs font-semibold">✓</div>
                          <div className="text-gray-500 text-xs">PTA</div>
                        </div>
                      </div>
                    </div>

                    {/* AI Features */}
                    <div className="space-y-3">
                      <div className="bg-gray-900 rounded-xl p-3 border border-[#f7f435]/30">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-[#f7f435]/20 rounded-lg flex items-center justify-center">
                            <span className="text-[#f7f435]">🤖</span>
                          </div>
                          <div className="flex-1">
                            <div className="text-white text-xs font-semibold">AI Damage Detection</div>
                            <div className="text-gray-500 text-xs">No cracks detected</div>
                          </div>
                          <div className="text-green-500 text-xs">✓</div>
                        </div>
                      </div>

                      <div className="bg-gray-900 rounded-xl p-3 border border-[#f7f435]/30">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-[#f7f435]/20 rounded-lg flex items-center justify-center">
                            <span className="text-[#f7f435]">💰</span>
                          </div>
                          <div className="flex-1">
                            <div className="text-white text-xs font-semibold">Price Prediction</div>
                            <div className="text-gray-500 text-xs">Market optimized</div>
                          </div>
                          <div className="text-green-500 text-xs">✓</div>
                        </div>
                      </div>

                      <div className="bg-gray-900 rounded-xl p-3 border border-[#f7f435]/30">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-[#f7f435]/20 rounded-lg flex items-center justify-center">
                            <span className="text-[#f7f435]">🔍</span>
                          </div>
                          <div className="flex-1">
                            <div className="text-white text-xs font-semibold">Sensor Diagnostics</div>
                            <div className="text-gray-500 text-xs">All tests passed</div>
                          </div>
                          <div className="text-green-500 text-xs">✓</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Glow Effect */}
                <div className="absolute -inset-4 bg-[#f7f435]/20 rounded-[4rem] blur-2xl -z-10"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Search Bar */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8 relative z-20">
        <SearchBar onSearch={handleSearch} />
      </section>

      {/* Featured Phones Carousel */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-3xl font-bold">Featured Phones</h2>
          <div className="flex gap-2">
            <button 
              onClick={prevSlide}
              className="p-3 rounded-lg yellow-btn"
              disabled={phones.length <= 3}
            >
              ◀
            </button>
            <button 
              onClick={nextSlide}
              className="p-3 rounded-lg yellow-btn"
              disabled={phones.length <= 3}
            >
              ▶
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {visiblePhones.map((phone) => (
            <ProductCard key={phone.id} phone={phone} />
          ))}
        </div>

        <div className="text-center">
          <Link 
            href="/marketplace"
            className="inline-block px-8 py-3 rounded-xl yellow-btn"
          >
            View All Phones
          </Link>
        </div>
      </section>

      {/* Why Choose IntelliFone */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl font-bold text-center mb-12">Why Choose IntelliFone?</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 text-center hover:scale-105 transition-transform">
            <div className="w-16 h-16 rounded-full bg-[#f7f435]/20 flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">✨</span>
            </div>
            <h3 className="text-xl font-bold mb-3">AI Damage Detection</h3>
            <p className="text-gray-400">
              Our advanced AI analyzes every inch of your phone to detect scratches, cracks, and hidden damage.
            </p>
          </div>

          <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10 p-8 text-center hover:scale-105 transition-transform">
            <div className="w-16 h-16 rounded-full bg-[#f7f435]/20 flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">📈</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Fair Price Prediction</h3>
            <p className="text-gray-400">
              Get accurate market prices based on real-time data, condition scores, and sensor diagnostics.
            </p>
          </div>

          <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 text-center hover:scale-105 transition-transform">
            <div className="w-16 h-16 rounded-full bg-[#f7f435]/20 flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🛡️</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Smart Recommendations</h3>
            <p className="text-gray-400">
              Find the perfect phone based on your budget, priorities, and usage patterns with AI recommendations.
            </p>
          </div>
        </div>
      </section>

      {/* Ready to Start Trading */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10 p-12 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-[#f7f435]/5"></div>
          <div className="relative z-10">
            <h2 className="text-4xl font-bold mb-4">Ready to Start Trading?</h2>
            <p className="text-xl text-gray-400 mb-8">
              List your phone in minutes and get instant AI verification
            </p>
            <Link 
              href="/sell"
              className="inline-block px-10 py-4 rounded-xl text-black text-lg font-semibold neon-glow"
              style={{ backgroundColor: "#f7f434" }}
            >
              Sell Your Phone
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}