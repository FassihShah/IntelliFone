// import { supabase } from "../../../lib/supabaseClient";
// import { NextResponse } from "next/server";

// export async function POST(req: Request) {
//   try {
//     // Parse JSON from frontend
//     const { formData, user_id, pictureUrls } = await req.json();

//     if (!pictureUrls || pictureUrls.length === 0) {
//       return NextResponse.json({ error: "No images provided" }, { status: 400 });
//     }

//     const fastapiUrl = "http://127.0.0.1:8000/damage-detection/"; // Replace with actual URL if deployed
//     const fastapiResponse = await fetch(fastapiUrl, {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({ image_urls: pictureUrls }),
//     });

//     if (!fastapiResponse.ok) {
//       const errText = await fastapiResponse.text();
//       return NextResponse.json(
//         { error: "Damage detection failed", details: errText },
//         { status: 500 }
//       );
//     }
//     const damagesHeader = fastapiResponse.headers.get("X-Damage-Results");
//     const damages = damagesHeader ? JSON.parse(damagesHeader) : {};



//     // Insert phone data into DB
//     const { data, error: dbError } = await supabase
//       .from("mobile_phones")
//       .insert({
//         ...formData,
//         user_id,
//         pictures: pictureUrls,
//       })
//       .select()
//       .single();

//     if (dbError) return NextResponse.json({ error: dbError.message }, { status: 400 });
//        console.log('Inserted phone data:', data);

//     return NextResponse.json({ success: true, id: data.id });
//   } catch (err) {
//     return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
//   }
// }


import { supabase } from "../../../lib/supabaseClient";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    // Parse JSON from frontend
    const { formData, user_id, pictureUrls } = await req.json();

    if (!pictureUrls || pictureUrls.length === 0) {
      return NextResponse.json({ error: "No images provided" }, { status: 400 });
    }

    // --- Step 1: Call FastAPI damage detection ---
    const fastapiUrl = "http://127.0.0.1:8000/damage-detection/";
    const fastapiResponse = await fetch(fastapiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_urls: pictureUrls }),
    });

    if (!fastapiResponse.ok) {
      const errText = await fastapiResponse.text();
      return NextResponse.json(
        { error: "Damage detection failed", details: errText },
        { status: 500 }
      );
    }

    // --- Step 2: Get PDF and damages ---
    const pdfBuffer = Buffer.from(await fastapiResponse.arrayBuffer());
    const damagesHeader = fastapiResponse.headers.get("X-Damage-Results");
    const damages = damagesHeader ? JSON.parse(damagesHeader) : {};

    // --- Step 3: Upload PDF to Supabase Storage ---
    const pdfName = `damage_reports/damage_report_${Date.now()}.pdf`;

    const { data: uploadData, error: uploadError } = await supabase.storage
      .from("phone-reports") // Your Supabase storage bucket
      .upload(pdfName, pdfBuffer, {
        contentType: "application/pdf",
        upsert: true,
      });

    if (uploadError) {
      return NextResponse.json({ error: uploadError.message }, { status: 500 });
    }

    // Generate public URL for PDF
    const { data: publicData } = supabase
      .storage
      .from("phone-reports")
      .getPublicUrl(pdfName);
    

    const publicUrl = publicData.publicUrl;

    // --- Step 4: Insert phone record in Supabase ---
    const { data, error: dbError } = await supabase
      .from("mobile_phones")
      .insert({
        ...formData,
        user_id,
        pictures: pictureUrls,
        damage_report_pdf: publicUrl,
      })
      .select()  
      .single();

    if (dbError)
      return NextResponse.json({ error: dbError.message }, { status: 400 });

    return NextResponse.json({
      success: true,
      id: data.id,
      damages,
      pdf_url: publicUrl,
    });

  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
