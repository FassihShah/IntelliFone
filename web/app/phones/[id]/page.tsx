// 'use client';

// import React, { useEffect, useState } from 'react';
// import Link from 'next/link';
// import { useParams, useRouter } from 'next/navigation';
// import ProductCard from '../../components/card/ProductCard';
// import {
//   ArrowLeft,
//   CheckCircle,
//   MapPin,
//   Zap,
//   FileText,
//   ShoppingCart,
//   MessageCircle,
//   Flag,
//   Star,
// } from 'lucide-react';

// /* 🔹 Phone Schema (as per parameter) */
// interface Phone {
//   id: string;
//   uuid: string;
//   model: string;
//   brand: string;
//   storage?: string;
//   ram?: string;
//   price: number;
//   images?: string[];
//   description?: string;
//   verified?: boolean;
//   condition_score?: number;
//   status?: string;
//   damage_report_pdf?: string;
// }

// export default function ProductDetailPage() {
//   const params = useParams();
//   const router = useRouter();
//   const phoneId = params.id as string;

//   const [phones, setPhones] = useState<Phone[]>([]);
//   const [phone, setPhone] = useState<Phone | null>(null);
//   const [currentImageIndex, setCurrentImageIndex] = useState(0);
//   const [loading, setLoading] = useState(true);

//   /* 🔹 Fetch phones from /api/list */
//   useEffect(() => {
//     async function fetchPhones() {
//       try {
//         const res = await fetch('/api/phones/list');
//         const data: Phone[] = await res.json();

//         setPhones(data);
//         const selected = data.find((p) => p.id === phoneId);
//         setPhone(selected ?? null);
//       } catch (err) {
//         console.error(err);
//       } finally {
//         setLoading(false);
//       }
//     }

//     if (phoneId) fetchPhones();
//   }, [phoneId]);

//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center">
//         <div className="w-12 h-12 border-4 border-[#f7f435] border-t-transparent rounded-full animate-spin"></div>
//       </div>
//     );
//   }

//   if (!phone) {
//     return (
//       <div className="min-h-screen flex items-center justify-center">
//         <div className="text-center">
//           <p className="text-2xl text-gray-400 mb-4">Phone not found</p>
//           <Link
//             href="/marketplace"
//             className="inline-block px-10 py-4 rounded-xl text-black text-lg font-semibold neon-glow px-6 py-3 rounded-lg inline-block"
//             style={{ backgroundColor: "#f7f434" }}
//           >
//             Back to Marketplace
//           </Link>
//         </div>
//       </div>
//     );
//   }

//   const images =
//     phone.images && phone.images.length > 0
//       ? phone.images
//       : ['https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600'];

//   const similarPhones = phones
//     .filter(
//       (p) =>
//         p.brand === phone.brand &&
//         p.id !== phone.id &&
//         p.status === 'active'
//     )
//     .slice(0, 4);

//   return (
//     <div className="min-h-screen py-8">
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

//         {/* 🔙 Back */}
//         <Link
//           href="/marketplace"
//           className="flex items-center gap-2 text-gray-400 hover:text-[#f7f435] mb-8 transition-colors"
//         >
//           <ArrowLeft className="w-4 h-4" />
//           Back to Marketplace
//         </Link>

//         {/* Product Layout */}
//         <div className="grid lg:grid-cols-2 gap-8 mb-16">

//           {/* Images */}
//           <div>
//             <div className="relative aspect-square bg-gray-900 rounded-2xl overflow-hidden">
//               {phone.verified && (
//                 <div className="absolute top-4 left-4 bg-[#f7f435] text-black px-3 py-1.5 rounded-lg flex items-center gap-2 text-sm font-semibold z-10">
//                   <CheckCircle className="w-4 h-4" />
//                   AI Verified
//                 </div>
//               )}
//               <img
//                 src={images[currentImageIndex]}
//                 alt={phone.model}
//                 className="w-full h-full object-cover"
//               />
//             </div>
//           </div>

//           {/* Details */}
//           <div className="space-y-6">

//             <div>
//               <h1 className="text-4xl font-bold mb-3">{phone.model}</h1>

//               <div className="flex items-center gap-3 text-sm text-gray-400 mb-4">
//                 <span> RAM : {phone.ram} GB </span>
//                 <span> Storage : {phone.storage} GB </span>
//                 <span>{phone.brand}</span>
//                 <span>Excellent</span>
//               </div>

//               <div className="text-4xl font-bold text-[#f7f435] mb-1">
//                 Rs. {phone.price.toLocaleString()}
//               </div>

//               <div className="flex items-center gap-2 text-sm text-gray-400 mt-3">
//                 <MapPin className="w-4 h-4" />
//                 <span>Pakistan</span>
//               </div>
//             </div>

//             {/* Description */}
//             {phone.description && (
//               <div className="glass-panel rounded-xl p-4">
//                 <h3 className="font-semibold mb-2">Description</h3>
//                 <p className="text-gray-400 text-sm leading-relaxed">
//                   {phone.description}
//                 </p>
//               </div>
//             )}

//             {/* Seller Info */}
//             <div className="glass-panel rounded-xl p-4">
//               <h3 className="font-semibold mb-3">Seller Information</h3>
//               <div className="flex items-center gap-3">
//                 <div className="w-12 h-12 bg-[#f7f435] rounded-full flex items-center justify-center text-black font-bold text-lg">
//                   D
//                 </div>
//                 <div>
//                   <div className="font-semibold">David Lee</div>
//                   <div className="flex items-center gap-1 text-sm text-gray-400">

//                   </div>
//                 </div>
//               </div>
//             </div>

//             {/* Actions */}
//             <div className="flex gap-3">
//               <Link
//                 href={`/checkout?id=${phone.id}`}
//                 className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-lg yellow-btn text-black font-semibold"
//                 style={{ backgroundColor: "#f7f434" }}
//               >
//                 <ShoppingCart className="w-5 h-5" />
//                 Buy Now
//               </Link>

//               <Link
//                 href={`/chat?id=${phone.id}`}
//                 className="px-6 py-3.5 rounded-lg glass-panel border border-gray-700 hover:border-[#f7f435] transition-colors flex items-center gap-2"
//               >
//                 <MessageCircle className="w-5 h-5" />
//                 Chat
//               </Link>

//               <Link
//                 href={`/report?id=${phone.id}`}
//                 className="px-6 py-3.5 rounded-lg glass-panel border border-gray-700 hover:border-red-500 transition-colors flex items-center justify-center"
//               >
//                 <Flag className="w-5 h-5" />
//               </Link>
//             </div>
//           </div>
//         </div>

//         {/* Similar Phones */}
//         {similarPhones.length > 0 && (
//           <section>
//             <h2 className="text-3xl font-bold mb-6">
//               Similar Phones <span className="text-[#f7f435]">You May Like</span>
//             </h2>

//             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
//               {similarPhones.map((p) => (
//                 <ProductCard key={p.id} phone={p} />
//               ))}
//             </div>
//           </section>
//         )}
//       </div>
//     </div>
//   );
// }

'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import ProductCard from '../../components/card/ProductCard';
import {
  ArrowLeft,
  CheckCircle,
  MapPin,
  Zap,
  FileText,
  ShoppingCart,
  MessageCircle,
  Flag,
  Star,
} from 'lucide-react';

/* 🔹 Phone Schema (as per parameter) */
interface Phone {
  id: string;
  uuid: string;
  model: string;
  brand: string;
  storage?: string;
  ram?: string;
  price: number;
  images?: string[];
  description?: string;
  verified?: boolean;
  condition_score?: number;
  status?: string;
  damage_report_pdf?: string;
}

export default function ProductDetailPage() {
  const params = useParams();
  const router = useRouter();
  const phoneId = params.id as string;

  const [phones, setPhones] = useState<Phone[]>([]);
  const [phone, setPhone] = useState<Phone | null>(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [loading, setLoading] = useState(true);

  /* 🔹 Fetch phones from /api/list */
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

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-[#f7f435] border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!phone) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-2xl text-gray-400 mb-4">Phone not found</p>
          <Link
            href="/marketplace"
            className="inline-block px-10 py-4 rounded-xl text-black text-lg font-semibold neon-glow px-6 py-3 rounded-lg inline-block"
            style={{ backgroundColor: "#f7f434" }}
          >
            Back to Marketplace
          </Link>
        </div>
      </div>
    );
  }

  const images =
    phone.images && phone.images.length > 0
      ? phone.images
      : ['https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600'];

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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* 🔙 Back */}
        <Link
          href="/marketplace"
          className="flex items-center gap-2 text-gray-400 hover:text-[#f7f435] mb-8 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Marketplace
        </Link>

        {/* Product Layout */}
        <div className="grid lg:grid-cols-2 gap-8 mb-16">

          {/* Images */}
          <div>
            <div className="relative aspect-square bg-gray-900 rounded-2xl overflow-hidden">
              {phone.verified && (
                <div className="absolute top-4 left-4 bg-[#f7f435] text-black px-3 py-1.5 rounded-lg flex items-center gap-2 text-sm font-semibold z-10">
                  <CheckCircle className="w-4 h-4" />
                  AI Verified
                </div>
              )}
              <img
                src={images[currentImageIndex]}
                alt={phone.model}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Details */}
          <div className="space-y-6">

            <div>
              <h1 className="text-4xl font-bold mb-3">{phone.model}</h1>

              <div className="flex items-center gap-3 text-sm text-gray-400 mb-4">
                <span> RAM : {phone.ram} GB </span>
                <span> Storage : {phone.storage} GB </span>
                <span>{phone.brand}</span>
                <span>Excellent</span>
              </div>

              <div className="text-4xl font-bold text-[#f7f435] mb-1">
                Rs. {phone.price.toLocaleString()}
              </div>

              <div className="flex items-center gap-2 text-sm text-gray-400 mt-3">
                <MapPin className="w-4 h-4" />
                <span>Pakistan</span>
              </div>
            </div>

            {/* Description */}
            {phone.description && (
              <div className="glass-panel rounded-xl p-4">
                <h3 className="font-semibold mb-2">Description</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  {phone.description}
                </p>
              </div>
            )}

            {/* Seller Info */}
            <div className="glass-panel rounded-xl p-4">
              <h3 className="font-semibold mb-3">Seller Information</h3>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-[#f7f435] rounded-full flex items-center justify-center text-black font-bold text-lg">
                  D
                </div>
                <div>
                  <div className="font-semibold">David Lee</div>
                  <div className="flex items-center gap-1 text-sm text-gray-400">

                  </div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <Link
                href={`/checkout?id=${phone.id}`}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-lg yellow-btn text-black font-semibold"
                style={{ backgroundColor: "#f7f434" }}
              >
                <ShoppingCart className="w-5 h-5" />
                Buy Now
              </Link>

              <Link
                href={`/chat?id=${phone.id}`}
                className="px-6 py-3.5 rounded-lg glass-panel border border-gray-700 hover:border-[#f7f435] transition-colors flex items-center gap-2"
              >
                <MessageCircle className="w-5 h-5" />
                Chat
              </Link>

              <Link
                href={`/report?id=${phone.id}`}
                className="px-6 py-3.5 rounded-lg glass-panel border border-gray-700 hover:border-red-500 transition-colors flex items-center justify-center"
              >
                <Flag className="w-5 h-5" />
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

