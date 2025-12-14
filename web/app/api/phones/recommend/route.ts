import { NextResponse } from 'next/server';

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);

    const max_price = searchParams.get('max_price');
    const priority = searchParams.get('priority');

    if (!max_price || !priority) {
      return NextResponse.json(
        { error: 'max_price and priority are required' },
        { status: 400 }
      );
    }

    // 🔁 Call FastAPI backend
    const response = await fetch(
      `http://127.0.0.1:8000/recommend/?max_price=${max_price}&priority=${encodeURIComponent(
        priority
      )}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        cache: 'no-store',
      }
    );

    if (!response.ok) {
      throw new Error('FastAPI request failed');
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: 'Failed to fetch recommendations' },
      { status: 500 }
    );
  }
}
