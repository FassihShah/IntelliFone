import React from 'react';
import { FileText, CheckCircle, AlertTriangle, Users, ShoppingCart, Shield, Scale, Ban } from 'lucide-react';

export default function TermsOfService() {
  const terms = [
    {
      icon: CheckCircle,
      title: 'Acceptance of Terms',
      content: 'By accessing and using IntelliFone, you accept and agree to be bound by the terms and provisions of this agreement. If you do not agree to these terms, please do not use our services.',
      color: 'green'
    },
    {
      icon: Users,
      title: 'User Accounts',
      content: 'You are responsible for maintaining the confidentiality of your account credentials and for all activities under your account. You must be at least 18 years old to create an account.',
      color: 'blue'
    },
    {
      icon: ShoppingCart,
      title: 'Buying & Selling',
      content: 'Sellers must provide accurate information about their phones. Buyers should review all product details and AI verification reports. All transactions are subject to our refund and return policy.',
      color: 'purple'
    },
    {
      icon: Shield,
      title: 'AI Verification',
      content: 'While our AI verification system is highly accurate, it should not be considered an absolute guarantee. We recommend inspecting phones in person before finalizing purchases.',
      color: 'yellow'
    },
    {
      icon: Ban,
      title: 'Prohibited Activities',
      content: 'Users may not engage in fraudulent activities, list counterfeit products, manipulate pricing, harass other users, or violate any applicable laws while using our platform.',
      color: 'red'
    },
    {
      icon: Scale,
      title: 'Dispute Resolution',
      content: 'Any disputes arising from transactions should first be resolved between buyers and sellers. IntelliFone may intervene to facilitate resolution but is not liable for transaction outcomes.',
      color: 'orange'
    },
    {
      icon: AlertTriangle,
      title: 'Limitation of Liability',
      content: 'IntelliFone is not liable for any indirect, incidental, or consequential damages arising from your use of the platform. Our total liability is limited to the amount paid for our services.',
      color: 'amber'
    },
    {
      icon: FileText,
      title: 'Changes to Terms',
      content: 'We reserve the right to modify these terms at any time. Continued use of the platform after changes constitutes acceptance of the modified terms.',
      color: 'cyan'
    }
  ];

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-[#f7f435]/20 mb-6">
            <FileText className="w-10 h-10 text-[#f7f435]" />
          </div>
          <h1 className="text-5xl font-bold mb-4">Terms of Service</h1>
          <p className="text-xl text-gray-400">Please read these terms carefully before using IntelliFone</p>
        </div>

        {/* Important Notice */}
        <div className="glass-panel rounded-2xl p-8 mb-12 border-2 border-[#f7f435]/30 bg-[#f7f435]/5">
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-8 h-8 text-[#f7f435] flex-shrink-0 mt-1" />
            <div>
              <h2 className="text-xl font-bold mb-2">Important Notice</h2>
              <p className="text-gray-300">
                These Terms of Service govern your use of IntelliFone and all related services. By using our platform, 
                you agree to comply with these terms and our Privacy Policy. Violations may result in account suspension or termination.
              </p>
            </div>
          </div>
        </div>

        {/* Terms Sections */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {terms.map((term, idx) => {
            const Icon = term.icon;
            return (
              <div key={idx} className="glass-panel rounded-2xl p-6 hover:scale-[1.02] transition-transform">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-12 h-12 rounded-xl bg-[#f7f435]/20 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-6 h-6 text-[#f7f435]" />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-lg font-bold text-white mb-2">{idx + 1}. {term.title}</h2>
                    <p className="text-gray-300 text-sm leading-relaxed">{term.content}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Agreement Section */}
        <div className="glass-panel rounded-2xl p-8 text-center border border-[#f7f435]/20 mb-8">
          <h2 className="text-2xl font-bold mb-4">Agreement & Acknowledgment</h2>
          <p className="text-gray-400 mb-6 max-w-2xl mx-auto">
            By using IntelliFone, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service 
            and our Privacy Policy. If you do not agree, you must discontinue use of the platform immediately.
          </p>
          <div className="flex justify-center gap-4">
            <a href="mailto:legal@intellifone.com" className="px-8 py-3 rounded-lg yellow-btn">
              Contact Legal Team
            </a>
          </div>
        </div>

        <p className="text-center text-sm text-gray-500">Last updated: January 2025</p>
      </div>
    </div>
  );
}