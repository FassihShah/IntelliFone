'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useRouter } from "next/navigation"

import ProductCard from '../../components/card/ProductCard';
import {
  ArrowLeft,
  CheckCircle,
  MapPin,
  ShoppingCart,
  MessageCircle,
  Flag,
  FileText,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { getOrCreateConversation } from '@/app/lib/chatService';
import { SupabaseClient } from '@supabase/supabase-js';
import { SupabaseAuthClient } from '@supabase/supabase-js/dist/module/lib/SupabaseAuthClient';
import { supabase } from '@/app/lib/supabaseClient';

/* 🔹 Phone Schema */
interface Phone {
  id: string;
  uuid:string
  user_id: string;
  model: string;
  brand: string;
  storage?: string;
  ram?: string;
  price: number;
  pictures?: string[];
  description?: string;
  verified?: boolean;
  condition_score?: number;
  status?: string;
  damage_report_pdf?: string;
}

export default function ProductDetailPage() {
  const params = useParams();
  const phoneId = params.id as string;

  const [phones, setPhones] = useState<Phone[]>([]);
  const [phone, setPhone] = useState<Phone | null>(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [sellerName, setSellerName] = useState('Loading...');
  const [sellerEmail, setSellerEmail] = useState('Loading...');

  /* 🔹 Fetch phones */
  useEffect(() => {
    async function fetchPhones() {
      try {
        const res = await fetch('/api/phones/list');
        const data: Phone[] = await res.json();

        setPhones(data);
        const selected = data.find((p) => p.id === phoneId);
        setPhone(selected ?? null);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    if (phoneId) fetchPhones();
  }, [phoneId]);

  /* 🔹 Fetch seller username */
  useEffect(() => {
    async function fetchSeller() {
      if (!phone?.user_id) return;

      try {
        const res = await fetch(`/api/users/${phone.user_id}`);

        console.log('Fetching seller info for user ID:', phone.user_id);
        const data = await res.json();
        setSellerName(data.username || 'Unknown Seller');
        setSellerEmail(data.email || 'No Email');
      } catch (err) {
        console.error(err);
        setSellerName('Unknown Seller');
      }
    }

    fetchSeller();
  }, [phone]);
  const [user, setUser] = useState<any>(null)

      useEffect(() => {
      supabase.auth.getUser().then(({ data }) => {
        setUser(data.user)
      })
    }, [])
  const router = useRouter()

async function handleContact(receiverId: string) {
  const currentUserId = user?.id;
  if (!currentUserId) return;

  // 1️⃣ Check if conversation already exists between A & B
  const { data: existingConvo } = await supabase
    .from("conversations")
    .select("*")
    .or(
      `user1_id.eq.${currentUserId},user2_id.eq.${receiverId}`
    )
    .single();

  let conversation = existingConvo

  // 2️⃣ If it doesn't exist, create it
  if (!conversation) {
    const { data: newConvo, error } = await supabase
      .from("conversation")
      .insert({
        user1_id: currentUserId,
        user2_id: receiverId,
      })
      .select()
      .single();

    if (error) {
      console.error("Failed to create conversation:", error)
      return
    }

    conversation = newConvo
  }

  // 3️⃣ Redirect
  router.push(`/chats?conversation=${conversation.id}`)
}



  /* 🔹 Add to Cart */
  function addToCart() {
    if (!phone) return;

    const cart = JSON.parse(localStorage.getItem('cart') || '[]');

    if (cart.find((item: any) => item.id === phone.id)) {
      alert('Already in cart');
      return;
    }

    cart.push({
      id: phone.id,
      model: phone.model,
      price: phone.price,
      image: images[0],
    });

    localStorage.setItem('cart', JSON.stringify(cart));
    alert('saved');
  }

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

  const images =
    phone.pictures?.length
      ? phone.pictures
      : ['https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600'];

  /* 🔹 Carousel controls */
  const nextImage = () =>
    setCurrentImageIndex((prev) => (prev + 1) % images.length);

  const prevImage = () =>
    setCurrentImageIndex(
      (prev) => (prev - 1 + images.length) % images.length
    );

  const similarPhones = phones
    .filter(
      (p) =>
        p.brand === phone.brand &&
        p.id !== phone.id &&
        p.status === 'active'
    )
    .slice(0, 4);

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-7xl mx-auto px-4">

        {/* Back */}
        <Link
          href="/marketplace"
          className="flex items-center gap-2 text-gray-400 hover:text-[#f7f435] mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Marketplace
        </Link>

        <div className="grid lg:grid-cols-2 gap-8 mb-16">

          {/* 🔹 IMAGE CAROUSEL */}
          <div>
            <div className="relative aspect-square bg-gray-900 rounded-2xl overflow-hidden">

              {phone.verified && (
                <div className="absolute top-4 left-4 bg-[#f7f435] text-black px-3 py-1.5 rounded-lg flex items-center gap-2 text-sm font-semibold z-10">
                  <CheckCircle className="w-4 h-4" />
                  AI Verified
                </div>
              )}

              {/* Main Image */}
              <img
                src={images[currentImageIndex]}
                alt={phone.model}
                className="w-full h-full object-cover"
              />

              {/* Arrows */}
              {images.length > 1 && (
                <>
                  <button
                    onClick={prevImage}
                    className="absolute left-3 top-1/2 -translate-y-1/2 bg-black/60 p-2 rounded-full"
                  >
                    <ChevronLeft />
                  </button>

                  <button
                    onClick={nextImage}
                    className="absolute right-3 top-1/2 -translate-y-1/2 bg-black/60 p-2 rounded-full"
                  >
                    <ChevronRight />
                  </button>
                </>
              )}
            </div>

            {/* Thumbnails */}
            {images.length > 1 && (
              <div className="flex gap-3 mt-4">
                {images.map((img, i) => (
                  <button
                    key={i}
                    onClick={() => setCurrentImageIndex(i)}
                    className={`w-20 h-20 rounded-xl overflow-hidden border ${
                      i === currentImageIndex
                        ? 'border-[#f7f435]'
                        : 'border-gray-700'
                    }`}
                  >
                    <img
                      src={img}
                      alt="thumbnail"
                      className="w-full h-full object-cover"
                    />
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Details */}
          <div className="space-y-6">
            <h1 className="text-4xl font-bold">{phone.model}</h1>

            <div className="flex gap-4 text-sm text-gray-400">
              <span>RAM: {phone.ram} GB</span>
              <span>Storage: {phone.storage} GB</span>
              <span>{phone.brand}</span>
            </div>

            <div className="text-4xl font-bold text-[#f7f435]">
              Rs. {phone.price.toLocaleString()}
            </div>

            <div className="flex items-center gap-2 text-sm text-gray-400">
              <MapPin className="w-4 h-4" />
              Pakistan
            </div>

            {phone.description && (
              <div className="glass-panel p-4 rounded-xl">
                {phone.description}
              </div>
            )}

            {/* Seller */}
            <div className="glass-panel p-4 rounded-xl">
              <h3 className="font-semibold mb-2">Seller</h3>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#f7f435] text-black rounded-full flex items-center justify-center font-bold">
                  {sellerName.charAt(0)}
                </div>
                <div>{sellerName}</div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap gap-3">
              

              <button
                onClick={addToCart}
                className="px-6 py-3.5 rounded-lg text-black font-semibold"
                 style={{ backgroundColor: '#f7f434' }}
              >
                Save
              </button>

              {phone.damage_report_pdf && (
                <Link
                  href={phone.damage_report_pdf}
                  target="_blank"
                  className="px-6 py-3.5 rounded-lg glass-panel border border-gray-700 flex items-center gap-2"
                >
                  <FileText className="w-5 h-5" />
                  View AI Report
                </Link>
              )}

              <button
                onClick={() => handleContact(phone.user_id)}
                className="px-6 py-3.5 rounded-lg glass-panel border border-yellow-500 hover:bg-yellow-500 hover:text-black transition"
              >
                <MessageCircle />
              </button>

              <Link
                href={`/report/${phone.id}`}
                className="px-6 py-3.5 rounded-lg glass-panel border border-gray-700 hover:border-red-500"
              >
                <Flag />
              </Link>
            </div>
          </div>
        </div>

        {/* Similar Phones */}
        {similarPhones.length > 0 && (
          <section>
            <h2 className="text-3xl font-bold mb-6">
              Similar Phones <span className="text-[#f7f435]">You May Like</span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {similarPhones.map((p) => (
                <ProductCard key={p.id} phone={p} />
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
