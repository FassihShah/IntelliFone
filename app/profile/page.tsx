'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '../lib/supabaseClient';
import Link from 'next/link';
import { Trash2, FileText } from 'lucide-react';

interface PhoneAd {
  id: string;
  user_id: string;
  model?: string;
  brand?: string;
  ram?: string;
  storage?: string;
  price?: number;
  pictures?: string[];
  status?: string;
  damage_report_pdf?: string;
  created_at?: string;
}

export default function ProfilePage() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);
  const [fullName, setFullName] = useState<string>('Loading...');
  const [ads, setAds] = useState<PhoneAd[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      if (!data?.user) {
        router.push('/signin');
        return;
      }
      setUser(data.user);
    });
  }, [router]);

  useEffect(() => {
    async function fetchProfileAndAds() {
      if (!user?.id) return;

      try {
        const profileRes = await fetch(`/api/users/${user.id}`);
        if (profileRes.ok) {
          const profile = await profileRes.json();
          setFullName(profile.full_name || 'Unknown User');
        } else {
          setFullName('Unknown User');
        }

        const adsRes = await fetch('/api/phones/list');
        const adsData: PhoneAd[] = await adsRes.json();
        setAds(adsData.filter((ad) => ad.user_id === user.id));
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchProfileAndAds();
  }, [user]);

  const handleDelete = async (adId: string) => {
    const confirmDelete = window.confirm(
      'Delete this ad? This will also remove its images and AI report.'
    );
    if (!confirmDelete || !user?.id) return;

    setDeletingId(adId);

    try {
      const res = await fetch(`/api/phones/delete/${adId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: user.id }),
      });

      const data = await res.json();
      if (!res.ok) {
        alert(data.error || 'Failed to delete ad.');
        return;
      }

      setAds((prev) => prev.filter((ad) => ad.id !== adId));
    } catch (err) {
      console.error(err);
      alert('Failed to delete ad.');
    } finally {
      setDeletingId(null);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-[#f7f435] border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-10 px-4 text-white">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="glass-panel rounded-2xl p-6 border border-gray-800">
          <h1 className="text-3xl font-bold mb-4">Your Profile</h1>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
            <div>
              <p className="text-gray-400">User ID</p>
              <p className="font-mono break-all">{user?.id}</p>
            </div>
            <div>
              <p className="text-gray-400">Name</p>
              <p className="text-lg font-semibold">{fullName}</p>
            </div>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Your Ads</h2>
            <Link
              href="/add"
              className="px-4 py-2 rounded-lg text-black font-semibold"
              style={{ backgroundColor: '#f7f434' }}
            >
              Post New Ad
            </Link>
          </div>

          {ads.length === 0 ? (
            <div className="glass-panel rounded-2xl p-6 border border-gray-800 text-gray-400">
              You have not posted any ads yet.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {ads.map((ad) => (
                <div
                  key={ad.id}
                  className="glass-panel rounded-2xl overflow-hidden border border-gray-800"
                >
                  <div className="relative aspect-square bg-gray-900">
                    <img
                      src={
                        ad.pictures?.[0] ||
                        'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400'
                      }
                      alt={ad.model || 'Phone'}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  <div className="p-4 space-y-2">
                    <h3 className="font-bold text-lg">
                      {ad.model || 'Untitled Phone'}
                    </h3>
                    <div className="text-[#f7f435] font-bold text-xl">
                      Rs. {ad.price?.toLocaleString() || 'N/A'}
                    </div>
                    <div className="flex gap-2 text-xs text-gray-400">
                      {ad.storage && (
                        <span className="bg-gray-800 px-2 py-1 rounded">
                          {ad.storage} GB
                        </span>
                      )}
                      {ad.ram && (
                        <span className="bg-gray-800 px-2 py-1 rounded">
                          {ad.ram} GB
                        </span>
                      )}
                      {ad.brand && (
                        <span className="bg-gray-800 px-2 py-1 rounded">
                          {ad.brand}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 mt-2">
                      <Link
                        href={`/phones/${ad.id}`}
                        className="flex-1 text-center py-2 rounded-lg text-black font-semibold"
                        style={{ backgroundColor: '#f7f434' }}
                      >
                        View
                      </Link>

                      {ad.damage_report_pdf && (
                        <Link
                          href={ad.damage_report_pdf}
                          target="_blank"
                          className="px-3 py-2 rounded-lg glass-panel border border-gray-700"
                        >
                          <FileText className="w-4 h-4" />
                        </Link>
                      )}

                      <button
                        onClick={() => handleDelete(ad.id)}
                        disabled={deletingId === ad.id}
                        className="px-3 py-2 rounded-lg bg-red-600 hover:bg-red-700 disabled:opacity-50"
                        title="Delete ad"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
