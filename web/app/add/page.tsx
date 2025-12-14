'use client';

import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabaseClient';
import { useRouter } from 'next/navigation';

export default function SellPhone() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);

  const [formData, setFormData] = useState({
    name: '',
    model: '',
    company: '',
    ram: '',
    storage: '',
    description: '',
    price: 0,
    pta_status: 'approved',
  });

  const [images, setImages] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);

  // Get logged-in user
  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      if (!data?.user) router.push('/signin');
      else setUser(data.user);
    });
  }, []);

  // Upload image to Supabase
  const uploadImages = async (e: any) => {
    const files = Array.from(e.target.files) as File[];
    if (files.length === 0) return;
    if (images.length + files.length > 6) {
      alert('You can upload max 6 images.');
      return;
    }

    setUploading(true);

    try {
      const uploadedURLs: string[] = [];

      for (const file of files) {
        const sanitizedFileName = file.name.replace(/\s+/g, "-").replace(/[^\w.-]/g, "");
        const fileName = `${Date.now()}-${sanitizedFileName}`;

        const { error } = await supabase.storage
          .from('phone-images')
          .upload(fileName, file);
        console.log('Upload error:', error);
        console.log("Logged-in user:", user);
        if (error) throw error;

        const { data: urlData } = supabase.storage
          .from('phone-images')
          .getPublicUrl(fileName);

        console.log('Uploaded URL:', urlData.publicUrl);

        uploadedURLs.push(urlData.publicUrl);
      }

      setImages(prev => [...prev, ...uploadedURLs]);
    } catch (err) {
      alert('Error uploading image.');
    }

    setUploading(false);
  };

  // Submit form
  const handleSubmit = async (e: any) => {
    e.preventDefault();
    if (!user) return;
    if (images.length < 1) {
      alert('Upload at least 1 image.');
      return;
    }

    const payload = {
      user_id: user.id,
      formData,
      pictureUrls: images, // send uploaded URLs
    };

    const res = await fetch('/api/phones/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.error || 'Something went wrong');
      return;
    }

    alert('Phone listed successfully!');
    router.push(`/products/${data.id}`);
   

  };

  return (
    <div className="min-h-screen py-8 px-4 text-white">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">Sell Your Mobile Phone</h1>

        <form onSubmit={handleSubmit} className="space-y-8">

          {/* IMAGE UPLOAD */}
          <div className="bg-gray-900/60 p-6 rounded-2xl border border-gray-700">
            <h2 className="text-xl font-bold mb-2">Upload Images</h2>
            <p className="text-gray-400 text-sm mb-3">Upload up to 6 images</p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {images.map((url, idx) => (
                <div key={idx} className="relative group rounded-xl overflow-hidden">
                  <img src={url} className="w-full h-full object-cover" />
                  <button
                    type="button"
                    onClick={() => setImages(images.filter((_, i) => i !== idx))}
                    className="absolute top-2 right-2 bg-red-600 p-1 rounded-full opacity-0 group-hover:opacity-100"
                  >
                    ❌
                  </button>
                </div>
              ))}

              {images.length < 6 && (
                <label className="border-2 border-dashed border-gray-700 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-yellow-400 transition">
                  <span className="text-3xl text-gray-300 mb-1">📤</span>
                  <span className="text-sm text-gray-400">Upload</span>
                  <input type="file" multiple accept="image/*" className="hidden" onChange={uploadImages} />
                </label>
              )}
            </div>

            {uploading && (
              <div className="mt-3 text-yellow-400 flex gap-2 items-center">
                <div className="w-4 h-4 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin"></div>
                Uploading...
              </div>
            )}
          </div>

          {/* DETAILS */}
          <div className="bg-gray-900/60 p-6 rounded-2xl border border-gray-700 space-y-4">
            <h2 className="text-xl font-bold mb-3">Phone Details</h2>

            <input className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="Phone Name"
              onChange={(e) => setFormData({ ...formData, name: e.target.value })} required />

            <input className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="Model"
              onChange={(e) => setFormData({ ...formData, model: e.target.value })} required />

            <input className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="Company"
              onChange={(e) => setFormData({ ...formData, company: e.target.value })} required />

            <input className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="RAM (e.g. 8)"
              onChange={(e) => setFormData({ ...formData, ram: e.target.value })} required />

            <input className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="Storage (e.g. 256)"
              onChange={(e) => setFormData({ ...formData, storage: e.target.value })} required />

            <input className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="Price (in PKR)"
              onChange={(e) => setFormData({ ...formData, storage: e.target.value })} required />

            <textarea className="w-full bg-black border border-gray-700 p-3 rounded-xl" placeholder="Description..."
              rows={4} onChange={(e) => setFormData({ ...formData, description: e.target.value })} />

            {/* PTA STATUS */}
            <div>
              <label className="text-sm">PTA Status</label>
              <select
                className="w-full bg-black border border-gray-700 p-3 rounded-xl mt-1"
                onChange={(e) => setFormData({ ...formData, pta_status: e.target.value })}
              >
                <option value="approved">Approved</option>
                <option value="non-approved">Not Approved</option>
              </select>
            </div>

          </div>

          <button
            type="submit"
            className="w-full bg-yellow-400 text-black py-3 rounded-xl font-bold text-lg hover:opacity-90"
          >
            Submit Listing
          </button>

        </form>
      </div>
    </div>
  );
}
