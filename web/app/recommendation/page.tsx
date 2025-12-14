"use client";

import React from 'react';
import { Sparkles, Battery, Camera, Gamepad2, Star, TrendingUp } from 'lucide-react';
import { Slider } from '@/components/ui/slider';

export default function Recommendations() {
  const [budgetRange, setBudgetRange] = React.useState([80000, 150000]);
  const [selectedPriority, setSelectedPriority] = React.useState('overall');
  const [aiResponse, setAiResponse] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [showResults, setShowResults] = React.useState(false);

  const priorities = [
    { id: 'camera', label: 'Camera', icon: Camera },
    { id: 'battery', label: 'Battery', icon: Battery },
    { id: 'gaming', label: 'Gaming', icon: Gamepad2 },
    { id: 'overall', label: 'Overall', icon: Star },
  ];

  const fetchAIRecommendations = async () => {
    setLoading(true);
    setAiResponse('');
    setShowResults(true);

    try {
      const res = await fetch(
        `/api/phones/recommend?max_price=${budgetRange[1]}&priority=${selectedPriority}`,
        { cache: 'no-store' }
      );

      const data = await res.json();
      const cleanData = JSON.stringify(data).replace(/\*/g, ' ');
      const parsedData = JSON.parse(cleanData);

      setAiResponse(parsedData.recommendations || 'No recommendation received.');
    } catch (error) {
      setAiResponse('Failed to get recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="mb-12">
          <h1 className="text-5xl font-bold mb-4">
            Find <span className="text-[#f7f435]">Best Phones</span> for You
          </h1>
          <p className="text-gray-400 text-lg">
            AI-powered phone recommendations based on your budget and priority
          </p>
        </div>

        {/* Preferences */}
        <div className="glass-panel rounded-2xl p-8 mb-8">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="w-5 h-5 text-[#f7f435]" />
            <h2 className="text-xl font-semibold">Your Preferences</h2>
          </div>

          {/* Budget */}
          <div className="mb-8">
            <label className="block font-semibold mb-4">
              Budget Range:{' '}
              <span className="text-[#f7f435]">
                Rs. {budgetRange[0].toLocaleString()} – Rs.{' '}
                {budgetRange[1].toLocaleString()}
              </span>
            </label>

            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <Slider
                min={10000}
                max={500000}
                step={10000}
                value={budgetRange}
                onValueChange={setBudgetRange}

              />
              <div className="flex justify-between text-sm text-gray-400 mt-4"
                   
              >
             
                <span>Rs. 10,000</span>
                <span>Rs. 500,000</span>
              </div>
            </div>
          </div>

          {/* Priority */}
          <div className="mb-8">
            <label className="block font-semibold mb-4">
              Select Your Main Priority
            </label>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {priorities.map((priority) => {
                const Icon = priority.icon;
                const active = selectedPriority === priority.id;

                return (
                  <button
                    key={priority.id}
                    onClick={() => setSelectedPriority(priority.id)}
                    className={`p-6 rounded-xl border-2 transition-all relative ${
                      active
                        ? 'border-[#f7f435] bg-[#f7f435]/10'
                        : 'border-gray-700 bg-gray-900/50 hover:border-gray-600'
                    }`}
                  >
                    {active && (
                      <div className="absolute top-2 right-2 w-5 h-5 bg-[#f7f435] rounded-full flex items-center justify-center">
                        <span className="text-black text-xs font-bold">✓</span>
                      </div>
                    )}

                    <Icon
                      className={`w-8 h-8 mx-auto mb-3 ${
                        active ? 'text-[#f7f435]' : 'text-gray-400'
                      }`}
                    />
                    <span className="block text-sm font-semibold">
                      {priority.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Button */}
          <button
            onClick={fetchAIRecommendations}
            disabled={loading}
            className="w-full py-4 rounded-xl yellow-btn text-black font-semibold flex items-center justify-center gap-2 disabled:opacity-60"
            style={{ backgroundColor: "#f7f434" }}
          >
            <Sparkles className="w-5 h-5" />
            {loading ? 'AI is thinking...' : 'Get AI Recommendation'}
          </button>
        </div>

        {/* AI Result */}
        {showResults && (
          <div className="glass-panel rounded-2xl p-8">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-[#f7f435]" />
              AI Recommendation
            </h2>

            {loading ? (
              <p className="text-gray-400 animate-pulse">
                Analyzing phones and generating recommendation...
              </p>
            ) : (
              <p className="text-gray-300 whitespace-pre-line leading-relaxed">
                {aiResponse}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
