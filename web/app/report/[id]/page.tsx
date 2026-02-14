'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter, useParams } from 'next/navigation';
import { ArrowLeft, Flag, AlertTriangle } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import emailjs from 'emailjs-com';

/* 🔹 EmailJS Config */
const SERVICE_ID = process.env.NEXT_PUBLIC_EMAILJS_SERVICE_ID!;
const TEMPLATE_ID = process.env.NEXT_PUBLIC_EMAILJS_TEMPLATE_ID!;
const PUBLIC_KEY = process.env.NEXT_PUBLIC_EMAILJS_PUBLIC_KEY!;

export default function ReportListing() {
  const params = useParams();
  const phoneId = params.id as string;
  const router = useRouter();

  const [reason, setReason] = useState('');
  const [details, setDetails] = useState('');
  const [phone, setPhone] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    async function fetchPhone() {
      try {
        const res = await fetch('/api/phones/list');
        const data = await res.json();
        const selected = data.find((p: any) => p.id === phoneId);
        setPhone(selected || null);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    if (phoneId) fetchPhone();
  }, [phoneId]);

  const reportReasons = [
    'Misleading information',
    'Fake/Counterfeit product',
    'Incorrect pricing',
    'Inappropriate content',
    'Scam or fraud',
    'Other',
  ];

  /* 🔴 EMAIL + SUBMIT HANDLER */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!reason || !details) {
      alert('Please select a reason and provide details.');
      return;
    }

    setIsSending(true);

    try {
      await emailjs.send(
        SERVICE_ID,
        TEMPLATE_ID,
        {
          name: phoneId,
          email: "tahakhan172004@gmail.com",
          subject: 'Listing Report Submission',
          message: reason + ' - ' + details + '' ,
        },
        PUBLIC_KEY
      );

      alert('Report submitted successfully. We will review it shortly.');
      router.push('/marketplace');
    } catch (error) {
      console.error(error);
      alert('Failed to submit report. Please try again.');
    } finally {
      setIsSending(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-[#f7f435] border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!phone) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400">
        Phone not found
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1 py-8 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <Link
          href={`/product/${phoneId}`}
          className="flex items-center gap-2 text-gray-400 hover:text-[#f7f435] mb-8 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Product
        </Link>

        <div className="flex items-center gap-3 mb-8">
          <Flag className="w-8 h-8 text-red-500" />
          <h1 className="text-4xl font-bold">Report Listing</h1>
        </div>

        <div className="glass-panel rounded-2xl p-6 mb-6">
          <h3 className="font-semibold mb-4">Reporting this listing:</h3>
          <div className="flex items-center gap-4">
            <img
              src={
                phone.pictures?.[0] ||
                'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400'
              }
              alt={phone.model}
              className="w-20 h-20 object-cover rounded-lg"
            />
            <div>
              <h4 className="font-semibold text-lg">{phone.model}</h4>
              <p className="text-gray-400">
                {phone.brand} • {phone.storage}
              </p>
              <p className="text-[#f7f435] font-bold">
                Rs. {phone.price?.toLocaleString()}
              </p>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="glass-panel rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="w-5 h-5 text-yellow-500" />
              <h2 className="text-xl font-bold">
                Why are you reporting this listing?
              </h2>
            </div>

            <div className="space-y-3 mb-6">
              {reportReasons.map((r) => (
                <label
                  key={r}
                  className="flex items-center gap-3 p-4 rounded-lg border border-gray-700 hover:border-red-500 cursor-pointer transition-colors"
                >
                  <input
                    type="radio"
                    name="reason"
                    value={r}
                    checked={reason === r}
                    onChange={(e) => setReason(e.target.value)}
                    className="w-4 h-4"
                    required
                  />
                  <span>{r}</span>
                </label>
              ))}
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Additional Details
              </label>
              <Textarea
                placeholder="Please provide more information about your report..."
                value={details}
                onChange={(e) => setDetails(e.target.value)}
                rows={6}
                className="bg-black border-gray-800"
                required
              />
            </div>
          </div>

          <div className="flex gap-4">
            <Link
              href={`/product/${phoneId}`}
              className="flex-1 px-6 py-3 rounded-lg glass-panel border border-gray-700 hover:border-gray-600 text-center transition-colors"
            >
              Cancel
            </Link>
            <button
              type="submit"
              disabled={isSending}
              className="flex-1 px-6 py-3 rounded-lg bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-semibold transition-colors"
            >
              {isSending ? 'Submitting...' : 'Submit Report'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
