'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { Trash2, Users, Smartphone, BarChart3 } from 'lucide-react';

interface UserRow {
  id: string;
  full_name?: string;
  email?: string;
  created_at?: string;
}

interface AdRow {
  id: string;
  user_id: string;
  model?: string;
  brand?: string;
  price?: number;
  created_at?: string;
}

type MonthlyPoint = { label: string; count: number };

function buildMonthlySeries(dates: Array<string | undefined>, months = 6): MonthlyPoint[] {
  const now = new Date();
  const points: MonthlyPoint[] = [];

  for (let i = months - 1; i >= 0; i -= 1) {
    const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
    const label = d.toLocaleString('en-US', { month: 'short' });
    points.push({ label, count: 0 });
  }

  for (const raw of dates) {
    if (!raw) continue;
    const d = new Date(raw);
    const diffMonths =
      (now.getFullYear() - d.getFullYear()) * 12 + (now.getMonth() - d.getMonth());
    if (diffMonths >= 0 && diffMonths < months) {
      const idx = months - diffMonths - 1;
      points[idx].count += 1;
    }
  }

  return points;
}

export default function AdminPage() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [ads, setAds] = useState<AdRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null);
  const [deletingAdId, setDeletingAdId] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAll() {
      try {
        const [usersRes, adsRes] = await Promise.all([
          fetch('/api/admin/users'),
          fetch('/api/admin/ads'),
        ]);

        const usersData = await usersRes.json();
        const adsData = await adsRes.json();

        setUsers(Array.isArray(usersData) ? usersData : []);
        setAds(Array.isArray(adsData) ? adsData : []);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchAll();
  }, []);

  const totalUsers = users.length;
  const totalAds = ads.length;
  const adsPerUser = totalUsers > 0 ? (totalAds / totalUsers).toFixed(2) : '0.00';

  const userSeries = useMemo(
    () => buildMonthlySeries(users.map((u) => u.created_at)),
    [users]
  );
  const adSeries = useMemo(
    () => buildMonthlySeries(ads.map((a) => a.created_at)),
    [ads]
  );

  const maxUserCount = Math.max(1, ...userSeries.map((p) => p.count));
  const maxAdCount = Math.max(1, ...adSeries.map((p) => p.count));

  const handleDeleteUser = async (userId: string) => {
    const ok = window.confirm(
      'Delete this user and all their ads? This also removes images and AI reports.'
    );
    if (!ok) return;

    setDeletingUserId(userId);
    try {
      const res = await fetch(`/api/admin/users/delete/${userId}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || 'Failed to delete user');
        return;
      }

      setUsers((prev) => prev.filter((u) => u.id !== userId));
      setAds((prev) => prev.filter((a) => a.user_id !== userId));
    } catch (err) {
      console.error(err);
      alert('Failed to delete user');
    } finally {
      setDeletingUserId(null);
    }
  };

  const handleDeleteAd = async (adId: string) => {
    const ok = window.confirm('Delete this ad and its files?');
    if (!ok) return;

    setDeletingAdId(adId);
    try {
      const res = await fetch(`/api/admin/ads/delete/${adId}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || 'Failed to delete ad');
        return;
      }

      setAds((prev) => prev.filter((a) => a.id !== adId));
    } catch (err) {
      console.error(err);
      alert('Failed to delete ad');
    } finally {
      setDeletingAdId(null);
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
      <div className="max-w-7xl mx-auto space-y-10">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-7 h-7 text-[#f7f435]" />
          <h1 className="text-3xl font-bold">Admin Dashboard</h1>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 text-gray-400 mb-2">
              <Users className="w-5 h-5" />
              Total Users
            </div>
            <div className="text-3xl font-bold">{totalUsers}</div>
          </div>
          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 text-gray-400 mb-2">
              <Smartphone className="w-5 h-5" />
              Total Ads
            </div>
            <div className="text-3xl font-bold">{totalAds}</div>
          </div>
          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 text-gray-400 mb-2">
              <BarChart3 className="w-5 h-5" />
              Ads / User
            </div>
            <div className="text-3xl font-bold">{adsPerUser}</div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <h2 className="text-xl font-bold mb-4">New Users (Last 6 Months)</h2>
            <div className="flex items-end gap-3 h-40">
              {userSeries.map((p) => (
                <div key={p.label} className="flex-1 flex flex-col items-center gap-2">
                  <div
                    className="w-full rounded-lg"
                    style={{
                      height: `${(p.count / maxUserCount) * 100}%`,
                      backgroundColor: '#f7f434',
                    }}
                  />
                  <span className="text-xs text-gray-400">{p.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <h2 className="text-xl font-bold mb-4">New Ads (Last 6 Months)</h2>
            <div className="flex items-end gap-3 h-40">
              {adSeries.map((p) => (
                <div key={p.label} className="flex-1 flex flex-col items-center gap-2">
                  <div
                    className="w-full rounded-lg"
                    style={{
                      height: `${(p.count / maxAdCount) * 100}%`,
                      backgroundColor: '#f7f434',
                    }}
                  />
                  <span className="text-xs text-gray-400">{p.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <h2 className="text-xl font-bold mb-4">Users</h2>
            <div className="space-y-3">
              {users.length === 0 && (
                <div className="text-gray-400">No users found.</div>
              )}
              {users.map((u) => (
                <div
                  key={u.id}
                  className="flex items-center justify-between gap-4 bg-black/40 p-3 rounded-xl border border-gray-800"
                >
                  <div>
                    <div className="font-semibold">
                      {u.full_name || 'Unknown User'}
                    </div>
                    <div className="text-sm text-gray-400">{u.email || '—'}</div>
                    <div className="text-xs text-gray-500 font-mono break-all">
                      {u.id}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteUser(u.id)}
                    disabled={deletingUserId === u.id}
                    className="px-3 py-2 rounded-lg bg-red-600 hover:bg-red-700 disabled:opacity-50"
                    title="Remove user"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-6 border border-gray-800">
            <h2 className="text-xl font-bold mb-4">Ads</h2>
            <div className="space-y-3">
              {ads.length === 0 && (
                <div className="text-gray-400">No ads found.</div>
              )}
              {ads.map((a) => (
                <div
                  key={a.id}
                  className="flex items-center justify-between gap-4 bg-black/40 p-3 rounded-xl border border-gray-800"
                >
                  <div>
                    <div className="font-semibold">
                      {a.model || 'Untitled'} {a.brand ? `• ${a.brand}` : ''}
                    </div>
                    <div className="text-sm text-gray-400">
                      Rs. {a.price?.toLocaleString() || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500 font-mono break-all">
                      {a.id}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Link
                      href={`/phones/${a.id}`}
                      className="px-3 py-2 rounded-lg glass-panel border border-gray-700"
                    >
                      View
                    </Link>
                    <button
                      onClick={() => handleDeleteAd(a.id)}
                      disabled={deletingAdId === a.id}
                      className="px-3 py-2 rounded-lg bg-red-600 hover:bg-red-700 disabled:opacity-50"
                      title="Remove ad"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
