'use client';

import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabaseClient';
import { useRouter } from 'next/navigation';

export default function SellPhone() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);

  const [formData, setFormData] = useState({
    company: '',
    model: '',
    ram: '',
    storage: ''
  });

  const [images, setImages] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [predicting, setPredicting] = useState(false);

  const [conditionScore, setConditionScore] = useState<number | null>(null);

  const [conditions, setConditions] = useState({
    is_panel_changed: false,
    screen_crack: false,
    panel_dot: false,
    panel_line: false,
    panel_shade: false,
    camera_lens_ok: true,
    fingerprint_ok: true,
    pta_approved: true
  });

  const [aiFlags, setAiFlags] = useState({
    screen_crack: false,
    panel_dot: false,
    panel_line: false
  });

  const [priceResult, setPriceResult] = useState<any>(null);

  /* -------------------- AUTH -------------------- */
  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      if (!data?.user) router.push('/signin');
      else setUser(data.user);
    });
  }, []);

  /* -------------------- IMAGE UPLOAD -------------------- */
  const uploadImages = async (e: any) => {
    const files = Array.from(e.target.files) as File[];
    if (files.length + images.length > 6) {
      alert('You can upload max 6 images.');
      return;
    }

    setUploading(true);
    const uploaded: string[] = [];

    for (const file of files) {
      const fileName = `${Date.now()}-${file.name.replace(/\s+/g, '-')}`;

      const { error } = await supabase.storage
        .from('phone-images')
        .upload(fileName, file);

      if (error) {
        alert('Upload failed');
        setUploading(false);
        return;
      }

      const { data } = supabase.storage
        .from('phone-images')
        .getPublicUrl(fileName);

      uploaded.push(data.publicUrl);
    }

    setImages(prev => [...prev, ...uploaded]);
    setUploading(false);
  };

  /* -------------------- AI DAMAGE DETECTION -------------------- */
  const analyzeImages = async () => {
    if (images.length === 0) {
      alert('Upload at least 1 image first.');
      return;
    }

    setAnalyzing(true);
    try {
      const res = await fetch('/api/damage-detection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_urls: images })
      });
      const data = await res.json();
      console.log('AI Damage Detection Result:', data);
      setConditionScore(data.condition_score || 0);
      setAiFlags(data.ai_detected || { screen_crack: false, panel_dot: false, panel_line: false });

      // Auto-fill only unchecked fields
      setConditions(prev => ({
        ...prev,
        screen_crack: prev.screen_crack || (data.ai_detected?.screen_crack ?? false),
        panel_dot: prev.panel_dot || (data.ai_detected?.panel_dot ?? false),
        panel_line: prev.panel_line || (data.ai_detected?.panel_line ?? false)
      }));
    } catch (err) {
      alert('Error analyzing images.');
    }
    setAnalyzing(false);
  };

  const predictPrice = async () => {
  if (!conditionScore) {
    alert('Analyze images first.');
    return;
  }

  setPredicting(true);

  try {
    const fd = new FormData();

    // Basic info
    fd.append('brand', formData.company);
    fd.append('model', formData.model);
    fd.append('ram', formData.ram);
    fd.append('storage', formData.storage);
    fd.append('condition_score', String(conditionScore));

    // User flags
    Object.entries(conditions).forEach(([key, value]) => {
      fd.append(key, String(value));
    });

    // AI flags
    Object.entries(aiFlags).forEach(([key, value]) => {
      fd.append(`ai_${key}`, String(value));
    });

    const res = await fetch('/api/price-prediction', {
      method: 'POST',
      body: fd
    });

    const data = await res.json();
    setPriceResult(data);
  } catch (err) {
    alert('Error predicting price.');
  }

  setPredicting(false);
};


  return (
    <div className="min-h-screen py-8 px-4 text-white">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold">Sell Your Mobile Phone</h1>

        {/* IMAGE UPLOAD */}
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
            <label className="border-2 border-dashed border-gray-700 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-yellow-400 transition p-6">
              <span className="text-3xl text-gray-300 mb-1">📤</span>
              <span className="text-sm text-gray-400">Drag & drop or click to upload</span>
              <input
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={uploadImages}
              />
            </label>
          )}
        </div>

        {images.length > 0 && (
          <button
            onClick={analyzeImages}
            disabled={analyzing}
            className="w-full mt-4 bg-yellow-400 text-black py-3 rounded-xl font-bold"
          >
            {analyzing ? 'Analyzing...' : 'Analyze Images'}
          </button>
        )}

        {/* PHONE DETAILS */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-700 space-y-3">
          <h2 className="text-xl font-bold">Phone Details</h2>

          <input
            className="w-full bg-black border border-gray-700 p-3 rounded-xl"
            placeholder="Brand"
            onChange={e => setFormData({ ...formData, company: e.target.value })}
          />
          <input
            className="w-full bg-black border border-gray-700 p-3 rounded-xl"
            placeholder="Model"
            onChange={e => setFormData({ ...formData, model: e.target.value })}
          />
          <input
            className="w-full bg-black border border-gray-700 p-3 rounded-xl"
            placeholder="RAM (GB)"
            onChange={e => setFormData({ ...formData, ram: e.target.value })}
          />
          <input
            className="w-full bg-black border border-gray-700 p-3 rounded-xl"
            placeholder="Storage (GB)"
            onChange={e => setFormData({ ...formData, storage: e.target.value })}
          />
        </div>

        {/* CONDITIONS */}
        {conditionScore !== null && (
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-700">
            <h2 className="text-xl font-bold mb-2">Condition Score: {conditionScore}</h2>
            <div className="grid grid-cols-2 gap-3">
              {Object.keys(conditions).map(key => (
                <label key={key} className="flex gap-2 items-center">
                  <input
                    type="checkbox"
                    checked={(conditions as any)[key]}
                    onChange={e =>
                      setConditions({ ...conditions, [key]: e.target.checked })
                    }
                  />
                  {key.replace(/_/g, ' ')}
                </label>
              ))}
            </div>

            <button
              onClick={predictPrice}
              disabled={predicting}
              className="w-full mt-4 bg-yellow-400 text-black py-3 rounded-xl font-bold"
            >
              {predicting ? 'Predicting...' : 'Get Price Estimate'}
            </button>
          </div>
        )}

        {/* PRICE RESULT */}
        {priceResult && (
          <div className="bg-green-900/30 border border-green-500 p-5 rounded-xl">
            <h3 className="text-xl font-bold mb-1">Estimated Price</h3>
            <p className="text-lg">{priceResult.min_price} – {priceResult.max_price} PKR</p>
          </div>
        )}
      </div>
    </div>
  );
}
