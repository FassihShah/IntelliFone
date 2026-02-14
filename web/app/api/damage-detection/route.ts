import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const res = await fetch('http://127.0.0.1:8000/damage-detection/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    const data = await res.json();

    return NextResponse.json({
      condition_score: data.condition_score,
      ai_detected: data.ai_detected
    });
  } catch (err) {
    return NextResponse.json(
      { error: 'AI detection failed' },
      { status: 500 }
    );
  }
}
