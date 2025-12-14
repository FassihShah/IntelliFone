import React from 'react';
import Link from 'next/link';

export default function About() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden py-20">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            About <span className="text-[#f7f435]">IntelliFone</span>
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Revolutionizing the used phone marketplace with AI-powered verification, 
            fair pricing, and smart recommendations.
          </p>
        </div>
      </section>

      {/* Mission */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="glass-panel rounded-2xl p-8 md:p-12">
          <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
          <p className="text-lg text-gray-400 leading-relaxed mb-4">
            At IntelliFone, we believe buying and selling used phones should be transparent, 
            trustworthy, and hassle-free. Traditional marketplaces lack proper verification, 
            leading to disputes and dissatisfaction.
          </p>
          <p className="text-lg text-gray-400 leading-relaxed">
            We leverage cutting-edge AI technology to analyze every device, detect damage, 
            verify sensor functionality, and provide accurate pricing—giving both buyers 
            and sellers complete confidence in every transaction.
          </p>
        </div>
      </section>

      {/* Why Different */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">Why IntelliFone is Different</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="glass-panel rounded-2xl p-8">
            <div className="w-14 h-14 rounded-full bg-[#f7f435]/20 flex items-center justify-center mb-4">
              <span className="text-2xl">✨</span>
            </div>
            <h3 className="text-xl font-bold mb-3">AI Damage Detection</h3>
            <p className="text-gray-400">
              Advanced computer vision analyzes every millimeter of your device to detect 
              scratches, cracks, and hidden damage invisible to the naked eye.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <div className="w-14 h-14 rounded-full bg-[#f7f435]/20 flex items-center justify-center mb-4">
              <span className="text-2xl">⚡</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Sensor Diagnostics</h3>
            <p className="text-gray-400">
              Comprehensive testing of all sensors including camera, microphone, GPS, 
              gyroscope, and more—ensuring full functionality before purchase.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <div className="w-14 h-14 rounded-full bg-[#f7f435]/20 flex items-center justify-center mb-4">
              <span className="text-2xl">📈</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Fair Price Prediction</h3>
            <p className="text-gray-400">
              Machine learning models analyze market data, condition scores, and sensor 
              health to provide accurate, fair pricing for every device.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <div className="w-14 h-14 rounded-full bg-[#f7f435]/20 flex items-center justify-center mb-4">
              <span className="text-2xl">🛡️</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Trust & Safety</h3>
            <p className="text-gray-400">
              Every listing is verified by AI before going live. Buyers can trust 
              the condition scores and sellers get fair evaluations.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <div className="w-14 h-14 rounded-full bg-[#f7f435]/20 flex items-center justify-center mb-4">
              <span className="text-2xl">👥</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Smart Recommendations</h3>
            <p className="text-gray-400">
              AI-powered recommendations help buyers find the perfect phone based on 
              their budget, priorities, and usage patterns.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <div className="w-14 h-14 rounded-full bg-[#f7f435]/20 flex items-center justify-center mb-4">
              <span className="text-2xl">🏆</span>
            </div>
            <h3 className="text-xl font-bold mb-3">Quality Guarantee</h3>
            <p className="text-gray-400">
              All verified devices come with detailed reports. Know exactly what you're 
              getting—no surprises, no hidden issues.
            </p>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">Our Values</h2>
        
        <div className="space-y-8">
          <div className="glass-panel rounded-2xl p-8">
            <h3 className="text-2xl font-bold mb-3">Transparency</h3>
            <p className="text-gray-400">
              Every device gets a complete condition report. No hidden details, 
              no misleading descriptions—just honest, AI-verified information.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <h3 className="text-2xl font-bold mb-3">Innovation</h3>
            <p className="text-gray-400">
              We continuously improve our AI models to provide better verification, 
              more accurate pricing, and smarter recommendations.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-8">
            <h3 className="text-2xl font-bold mb-3">Customer First</h3>
            <p className="text-gray-400">
              Your satisfaction is our priority. We build tools that make buying and 
              selling phones easier, safer, and more reliable for everyone.
            </p>
          </div>
        </div>
      </section>

      {/* User Reviews */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl font-bold text-center mb-4">What Our Users Say</h2>
        <p className="text-center text-gray-400 mb-12">Trusted by thousands of buyers and sellers</p>
        
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {[
            {
              name: 'Ahmed Khan',
              role: 'Seller',
              rating: 5,
              review: 'IntelliFone made selling my phone so easy! The AI verification gave buyers confidence and I sold within 2 days.',
              image: 'A'
            },
            {
              name: 'Sarah Ahmed',
              role: 'Buyer',
              rating: 5,
              review: 'Best marketplace for used phones. The AI reports are incredibly detailed and helped me make an informed decision.',
              image: 'S'
            },
            {
              name: 'Hassan Ali',
              role: 'Seller',
              rating: 4,
              review: 'Great platform with fair pricing. The sensor diagnostics feature is a game-changer for building trust with buyers.',
              image: 'H'
            }
          ].map((review, idx) => (
            <div key={idx} className="glass-panel rounded-2xl p-6 hover:scale-105 transition-transform">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-12 h-12 rounded-full bg-[#f7f435] flex items-center justify-center text-black font-bold text-xl">
                  {review.image}
                </div>
                <div>
                  <h4 className="font-semibold">{review.name}</h4>
                  <p className="text-sm text-gray-400">{review.role}</p>
                </div>
              </div>
              <div className="flex gap-1 mb-3">
                {[...Array(review.rating)].map((_, i) => (
                  <span key={i} className="text-[#f7f435]">★</span>
                ))}
              </div>
              <p className="text-gray-300 text-sm">{review.review}</p>
            </div>
          ))}
        </div>

        <div className="glass-panel rounded-2xl p-8">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <p className="text-4xl font-bold text-[#f7f435] mb-2">10K+</p>
              <p className="text-gray-400">Happy Users</p>
            </div>
            <div>
              <p className="text-4xl font-bold text-[#f7f435] mb-2">5K+</p>
              <p className="text-gray-400">Phones Sold</p>
            </div>
            <div>
              <p className="text-4xl font-bold text-[#f7f435] mb-2">4.8</p>
              <p className="text-gray-400">Average Rating</p>
            </div>
            <div>
              <p className="text-4xl font-bold text-[#f7f435] mb-2">98%</p>
              <p className="text-gray-400">Satisfaction Rate</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="glass-panel rounded-2xl p-12 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-[#f7f435]/5"></div>
          <div className="relative z-10">
            <h2 className="text-4xl font-bold mb-4">Ready to Experience the Difference?</h2>
            <p className="text-xl text-gray-400 mb-8">
              Join thousands of users who trust IntelliFone for their phone trading needs
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <Link href="/marketplace" className="px-8 py-4 rounded-xl yellow-btn text-lg">
                Browse Marketplace
              </Link>
              <Link href="/sell" className="px-8 py-4 rounded-xl glass-panel border border-[#f7f435] hover:bg-[#f7f435] hover:text-black transition-all text-lg">
                Sell Your Phone
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}