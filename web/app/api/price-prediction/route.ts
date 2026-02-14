import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const body = await req.formData(); // receive as FormData from frontend

    const fd = new FormData();

    // Map all fields to FastAPI
    for (const [key, value] of body.entries()) {
      fd.append(key, String(value));
    }

    const res = await fetch('http://127.0.0.1:8000/price-prediction/', {
      method: 'POST',
      body: fd
    });

    const data = await res.json();

    return NextResponse.json(data);
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: 'Price prediction failed' }, { status: 500 });
  }
}
