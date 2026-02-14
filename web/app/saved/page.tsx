'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { Trash2, ShoppingCart, Heart } from 'lucide-react';

interface SavedPhone {
  id: string;
  model: string;
  price: number;
  image: string;
}

export default function SavedPhonesPage() {
  const [phones, setPhones] = useState<SavedPhone[]>([]);

  useEffect(() => {
    const cart = JSON.parse(localStorage.getItem('cart') || '[]');
    setPhones(cart);
  }, []);

  const removePhone = (id: string) => {
    const updated = phones.filter((p) => p.id !== id);
    setPhones(updated);
    localStorage.setItem('cart', JSON.stringify(updated));
  };

  if (phones.length === 0) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-center">
        <Heart className="w-14 h-14 text-gray-500 mb-4" />
        <h1 className="text-2xl font-bold mb-2">No Saved Phones</h1>
        <p className="text-gray-400 mb-6">
          Phones you save will appear here.
        </p>
        <Link
          href="/marketplace"
          className="px-6 py-3 rounded-lg bg-[#f7f435] text-black font-semibold"
        >
          Browse Marketplace
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 sm:px-6 lg:px-8 py-10 max-w-7xl mx-auto">
      <div className="flex items-center gap-3 mb-8">
        <ShoppingCart className="w-7 h-7 text-[#f7f435]" />
        <h1 className="text-4xl font-bold">Saved Phones</h1>
      </div>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {phones.map((phone) => (
          <div
            key={phone.id}
            className="glass-panel rounded-2xl overflow-hidden border border-gray-800 hover:border-[#f7f435] transition"
          >
            <img
              src={phone.image}
              alt={phone.model}
              className="w-full h-52 object-cover"
            />

            <div className="p-5 space-y-3">
              <h3 className="text-xl font-semibold">{phone.model}</h3>

              <p className="text-[#f7f435] text-lg font-bold">
                Rs. {phone.price.toLocaleString()}
              </p>

              <div className="flex gap-3 pt-3">
                <Link
                  href={`/product/${phone.id}`}
                  className="flex-1 text-center px-4 py-2 rounded-lg bg-[#f7f435] text-black font-semibold hover:opacity-90"
                >
                  View
                </Link>

                <button
                  onClick={() => removePhone(phone.id)}
                  className="px-4 py-2 rounded-lg border border-red-600 text-red-500 hover:bg-red-600 hover:text-white transition"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
