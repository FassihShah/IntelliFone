import { supabase } from "@/app/lib/supabaseClient";
import { NextResponse } from "next/server";
import fs from "fs";

export async function POST(req: Request) {
  try {
    const { formData, user_id, pictureUrls } = await req.json();

    if (!pictureUrls?.length) {
      return NextResponse.json({ error: "No images provided" }, { status: 400 });
    }

    // 1️⃣ Call FastAPI
    const fastapiRes = await fetch("http://127.0.0.1:8000/damage-detection/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_urls: pictureUrls }),
    });

    const fastapiData = await fastapiRes.json();
    console.log("FastAPI response:", fastapiData);

    const { pdf_path, condition_score } = fastapiData;
    if (!fastapiData.pdf_path) {
      return NextResponse.json({ error: "PDF path missing" }, { status: 500 });
    }

    // // 2️⃣ Read PDF from disk
    // const pdfBuffer = fs.readFileSync(fastapiData.pdf_path);

    // // 3️⃣ Upload to Supabase
    // const pdfName = `damage_reports/report_${Date.now()}.pdf`;

    // const { error: uploadError } = await supabase.storage
    //   .from("phone-reports")
    //   .upload(pdfName, pdfBuffer, {
    //     contentType: "application/pdf",
    //     upsert: true,
    //   });

    // if (uploadError) {
    //   return NextResponse.json({ error: uploadError.message }, { status: 500 });
    // }

    

    console.log(formData.price);

    // 4️⃣ Save DB record
    const { data, error: dbError } = await supabase
      .from("mobile_phones")
      .insert({
        ...formData,
        user_id,
        pictures: pictureUrls,
        condition_score: condition_score,
        damage_report_pdf: pdf_path, // Store the local path for now
      })
      .select()
      .single();

    if (dbError) {
      return NextResponse.json({ error: dbError.message }, { status: 400 });
    }

    return NextResponse.json({
      success: true,
      id: data.id,
      pdf_url: pdf_path, // Return the local path for now
    });

  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
