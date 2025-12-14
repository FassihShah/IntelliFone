import React from 'react';
import Link from "next/link";
import { createPageUrl } from  "./../../lib/utils";
import { HelpCircle, Search, Phone, Mail, MessageCircle, BookOpen } from 'lucide-react';
import { Input } from '@/components/ui/input';

export default function HelpCenter() {
  const [searchQuery, setSearchQuery] = React.useState('');

  const faqs = [
    {
      category: 'Buying',
      questions: [
        { q: 'How do I buy a phone?', a: 'Browse our marketplace, select a phone, and click Buy Now to proceed with checkout.' },
        { q: 'Is the AI verification reliable?', a: 'Yes, our AI uses advanced image analysis to detect damage with high accuracy.' },
        { q: 'What payment methods are accepted?', a: 'We accept Cash on Delivery and online payments via credit/debit cards.' },
      ]
    },
    {
      category: 'Selling',
      questions: [
        { q: 'How do I list my phone?', a: 'Go to the Sell page, upload images, fill in details, and submit your listing.' },
        { q: 'How long does verification take?', a: 'AI verification is instant. Manual review may take 1-2 business days.' },
        { q: 'What are the selling fees?', a: 'We charge a small 5% commission on successful sales.' },
      ]
    },
    {
      category: 'Account',
      questions: [
        { q: 'How do I create an account?', a: 'Click Sign Up in the header and follow the registration process.' },
        { q: 'Can I edit my listing?', a: 'Yes, go to your Dashboard and click Edit on any of your active listings.' },
        { q: 'How do I contact support?', a: 'Visit our Contact page or email support@intellifone.com.' },
      ]
    }
  ];

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <HelpCircle className="w-12 h-12 text-[#f7f435]" />
            <h1 className="text-5xl font-bold">Help Center</h1>
          </div>
          <p className="text-xl text-gray-400">How can we help you today?</p>
        </div>

        {/* Search */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <Input
              placeholder="Search for help..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-12 py-6 text-lg bg-gray-900 border-gray-800"
            />
          </div>
        </div>

        {/* Quick Contact */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <Link href={createPageUrl('Contact')} className="glass-panel rounded-2xl p-6 hover:scale-105 transition-transform text-center">
            <Phone className="w-8 h-8 text-[#f7f435] mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Call Us</h3>
            <p className="text-sm text-gray-400">+92 123 456 7890</p>
          </Link>
          <Link href={createPageUrl('Contact')} className="glass-panel rounded-2xl p-6 hover:scale-105 transition-transform text-center">
            <Mail className="w-8 h-8 text-[#f7f435] mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Email Us</h3>
            <p className="text-sm text-gray-400">support@intellifone.com</p>
          </Link>
          <div className="glass-panel rounded-2xl p-6 hover:scale-105 transition-transform text-center cursor-pointer">
            <MessageCircle className="w-8 h-8 text-[#f7f435] mx-auto mb-3" />
            <h3 className="font-semibold mb-2">Live Chat</h3>
            <p className="text-sm text-gray-400">Chat with support</p>
          </div>
        </div>

        {/* FAQs */}
        <div className="space-y-8">
          {faqs.map((category) => (
            <div key={category.category}>
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <BookOpen className="w-6 h-6 text-[#f7f435]" />
                {category.category}
              </h2>
              <div className="space-y-4">
                {category.questions.map((faq, idx) => (
                  <div key={idx} className="glass-panel rounded-xl p-6">
                    <h3 className="font-semibold mb-2">{faq.q}</h3>
                    <p className="text-gray-400">{faq.a}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}