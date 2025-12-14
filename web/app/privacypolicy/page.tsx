import React from 'react';
import { Shield, Lock, Eye, UserCheck, Database, Bell, Globe } from 'lucide-react';

export default function PrivacyPolicy() {
  const sections = [
    {
      icon: Database,
      title: 'Information We Collect',
      content: 'We collect information you provide directly to us when you create an account, list a phone, or contact us. This includes your name, email address, phone number, and payment information. We also collect device information and usage data to improve our services.'
    },
    {
      icon: Lock,
      title: 'How We Use Your Information',
      content: 'We use the information we collect to provide, maintain, and improve our services, process transactions, send you technical notices and support messages, respond to your comments and questions, and provide personalized recommendations.'
    },
    {
      icon: UserCheck,
      title: 'Information Sharing',
      content: 'We do not share your personal information with third parties except as described in this policy. We may share information with vendors and service providers who perform services on our behalf, and only when necessary to provide our services.'
    },
    {
      icon: Shield,
      title: 'Data Security',
      content: 'We implement appropriate security measures to protect your personal information, including encryption, secure servers, and regular security audits. However, no method of transmission over the internet is 100% secure.'
    },
    {
      icon: Eye,
      title: 'Your Rights',
      content: 'You have the right to access, update, or delete your personal information. You can also opt-out of marketing communications, request a copy of your data, and control your privacy settings through your account dashboard.'
    },
    {
      icon: Bell,
      title: 'Cookies & Tracking',
      content: 'We use cookies and similar technologies to improve your experience, analyze usage patterns, and provide personalized content. You can control cookie settings through your browser preferences.'
    },
    {
      icon: Globe,
      title: 'International Transfers',
      content: 'Your information may be transferred to and processed in countries other than your own. We ensure appropriate safeguards are in place to protect your data in accordance with this policy.'
    }
  ];

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-[#f7f435]/20 mb-6">
            <Shield className="w-10 h-10 text-[#f7f435]" />
          </div>
          <h1 className="text-5xl font-bold mb-4">Privacy Policy</h1>
          <p className="text-xl text-gray-400">Your privacy is our priority. Learn how we protect and use your data.</p>
        </div>

        {/* Quick Summary */}
        <div className="glass-panel rounded-2xl p-8 mb-12 border border-[#f7f435]/20">
          <h2 className="text-2xl font-bold mb-4">Quick Summary</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center p-4">
              <Lock className="w-8 h-8 text-[#f7f435] mx-auto mb-3" />
              <h3 className="font-semibold mb-2">Secure Data</h3>
              <p className="text-sm text-gray-400">Your data is encrypted and stored securely</p>
            </div>
            <div className="text-center p-4">
              <UserCheck className="w-8 h-8 text-[#f7f435] mx-auto mb-3" />
              <h3 className="font-semibold mb-2">Your Control</h3>
              <p className="text-sm text-gray-400">You control your data and privacy settings</p>
            </div>
            <div className="text-center p-4">
              <Eye className="w-8 h-8 text-[#f7f435] mx-auto mb-3" />
              <h3 className="font-semibold mb-2">Transparent</h3>
              <p className="text-sm text-gray-400">Clear about how we use your information</p>
            </div>
          </div>
        </div>

        {/* Detailed Sections */}
        <div className="space-y-6 mb-12">
          {sections.map((section, idx) => {
            const Icon = section.icon;
            return (
              <div key={idx} className="glass-panel rounded-2xl p-6 hover:scale-[1.02] transition-transform">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-xl bg-[#f7f435]/20 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-6 h-6 text-[#f7f435]" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white mb-3">{idx + 1}. {section.title}</h2>
                    <p className="text-gray-300 leading-relaxed">{section.content}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Contact Section */}
        <div className="glass-panel rounded-2xl p-8 text-center border border-[#f7f435]/20">
          <h2 className="text-2xl font-bold mb-4">Have Questions?</h2>
          <p className="text-gray-400 mb-6">If you have any questions about this Privacy Policy, please don't hesitate to contact us.</p>
          <a href="mailto:privacy@intellifone.com" className="inline-block px-8 py-3 rounded-lg yellow-btn">
            Contact Privacy Team
          </a>
        </div>

        <p className="text-center text-sm text-gray-500 mt-8">Last updated: January 2025</p>
      </div>
    </div>
  );
}