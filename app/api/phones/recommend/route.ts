import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
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

    // 🔁 Call FastAPI streaming endpoint
    const response = await fetch(
      `http://127.0.0.1:8000/recommend-stream/?max_price=${max_price}&priority=${encodeURIComponent(
        priority
      )}`,
      {
        method: 'GET',
        headers: { Accept: 'text/plain' },
        cache: 'no-store',
      }
    );

    if (!response.ok || !response.body) {
      throw new Error('FastAPI streaming request failed');
    }

    // Pipe the FastAPI stream straight back to the browser
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: 'Failed to fetch recommendations' },
      { status: 500 }
    );
  }
}
