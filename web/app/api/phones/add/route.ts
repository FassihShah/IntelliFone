import { supabase } from "@/app/lib/supabaseClient";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { formData, user_id, pictureUrls } = await req.json();

    if (!pictureUrls?.length) {
      return NextResponse.json({ error: "No images provided" }, { status: 400 });
    }

    const fastapiBaseUrl = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
    const fastapiRes = await fetch(`${fastapiBaseUrl}/damage-detection/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_urls: pictureUrls }),
    });

    const fastapiData = await fastapiRes.json();
    console.log("FastAPI response:", fastapiData);

    if (!fastapiRes.ok) {
      return NextResponse.json(
        { error: fastapiData?.detail || fastapiData?.error || "FastAPI request failed" },
        { status: fastapiRes.status }
      );
    }

    const { pdf_url, condition_score } = fastapiData;
    if (!pdf_url) {
      return NextResponse.json({ error: "PDF URL missing" }, { status: 500 });
    }

    const { data, error: dbError } = await supabase
      .from("mobile_phones")
      .insert({
        ...formData,
        user_id,
        pictures: pictureUrls,
        condition_score,
        damage_report_pdf: pdf_url,
      })
      .select()
      .single();

    if (dbError) {
      return NextResponse.json({ error: dbError.message }, { status: 400 });
    }

    return NextResponse.json({
      success: true,
      id: data.id,
      pdf_url,
    });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
