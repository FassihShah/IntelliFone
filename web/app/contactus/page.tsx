"use client";

import React from "react";
import emailjs from "emailjs-com";
import { Mail, MessageCircle, Send, MapPin, Phone } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export default function Contact() {
  const [formData, setFormData] = React.useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });

  const [isSending, setIsSending] = React.useState(false);
  const [submitted, setSubmitted] = React.useState(false);

  // ✅ env variables (correct way)
  const SERVICE_ID = process.env.NEXT_PUBLIC_EMAILJS_SERVICE_ID!;
  const TEMPLATE_ID = process.env.NEXT_PUBLIC_EMAILJS_TEMPLATE_ID!;
  const PUBLIC_KEY = process.env.NEXT_PUBLIC_EMAILJS_PUBLIC_KEY!;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSending(true);

    try {
      await emailjs.send(
        SERVICE_ID,
        TEMPLATE_ID,
        {
          name: formData.name,
          email: formData.email,
          subject: formData.subject,
          message: formData.message,
        },
        PUBLIC_KEY
      );

      setSubmitted(true);
      setFormData({ name: "", email: "", subject: "", message: "" });
    } catch (error) {
      alert("Failed to send message. Please try again.");
    }

    setIsSending(false);
  };

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Get in Touch</h1>
          <p className="text-xl text-gray-400">
            Have questions? We'd love to hear from you.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10">
            <h2 className="text-2xl font-bold mb-6">Send us a Message</h2>

            {submitted ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
                  <Mail className="w-8 h-8 text-green-500" />
                </div>
                <h3 className="text-xl font-bold mb-2">Message Sent!</h3>
                <p className="text-gray-400 mb-6">
                  Thank you for contacting us. We'll get back to you within 24 hours.
                </p>
                <button
                  onClick={() => setSubmitted(false)}
                  className="px-6 py-3 rounded-lg yellow-btn"
                >
                  Send Another Message
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2">Your Name</label>
                  <Input
                    value={formData.name}
                    onChange={(e) =>
                      setFormData({ ...formData, name: e.target.value })
                    }
                    placeholder="John Doe"
                    required
                    className="bg-black border-gray-800"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Email Address</label>
                  <Input
                    type="email"
                    value={formData.email}
                    onChange={(e) =>
                      setFormData({ ...formData, email: e.target.value })
                    }
                    placeholder="john@example.com"
                    required
                    className="bg-black border-gray-800"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Subject</label>
                  <Input
                    value={formData.subject}
                    onChange={(e) =>
                      setFormData({ ...formData, subject: e.target.value })
                    }
                    placeholder="How can we help?"
                    required
                    className="bg-black border-gray-800"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Message</label>
                  <Textarea
                    value={formData.message}
                    onChange={(e) =>
                      setFormData({ ...formData, message: e.target.value })
                    }
                    placeholder="Tell us more about your inquiry..."
                    rows={6}
                    required
                    className="bg-black border-gray-800"
                  />
                </div>

               <button
                type="submit"
                disabled={isSending}
                className="w-full px-6 py-3 rounded-lg text-black font-semibold flex items-center justify-center gap-2 disabled:opacity-50 transition"
                style={{
                    backgroundColor: "#f7f434",
                }}
                >
                {isSending ? (
                    <>
                    <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                    Sending...
                    </>
                ) : (
                    <>
                    <Send className="w-5 h-5" />
                    Send Message
                    </>
                )}
                </button>

              </form>
            )}
          </div>

          {/* Contact Information (UNCHANGED) */}
          <div className="space-y-8">
            <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10">
              <h2 className="text-2xl font-bold mb-6">Contact Information</h2>

              <div className="space-y-6">
                <Info icon={<Mail />} title="Email" text="support@intellifone.com" />
                <Info icon={<Phone />} title="Phone" text="+92 300 1234567" />
                <Info icon={<MapPin />} title="Location" text="Karachi, Pakistan" />
                <Info icon={<MessageCircle />} title="Live Chat" text="Available on marketplace" />
              </div>
            </div>
            <div className="rounded-2xl p-8 bg-white/5 backdrop-blur-xl border border-white/10">
              <h3 className="text-xl font-bold mb-4">Frequently Asked Questions</h3>
              <div className="space-y-4 text-sm">
                <div>
                  <p className="font-semibold mb-1">How long does AI verification take?</p>
                  <p className="text-gray-400">Usually 2-5 minutes after uploading images</p>
                </div>
                <div>
                  <p className="font-semibold mb-1">Is the price prediction accurate?</p>
                  <p className="text-gray-400">Our AI analyzes market data with 85-95% accuracy</p>
                </div>
                <div>
                  <p className="font-semibold mb-1">How do I verify my phone?</p>
                  <p className="text-gray-400">Upload 6 images from different angles for AI analysis</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
  );
}

function Info({ icon, title, text }: any) {
  return (
    <div className="flex items-start gap-4">
      <div className="w-12 h-12 rounded-full bg-[#f7f435]/20 flex items-center justify-center">
        <div className="text-[#f7f435]">{icon}</div>
      </div>
      <div>
        <h3 className="font-semibold mb-1">{title}</h3>
        <p className="text-gray-400">{text}</p>
      </div>
    </div>
  );
}
