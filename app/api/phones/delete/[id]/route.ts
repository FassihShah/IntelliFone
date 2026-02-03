import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/app/lib/supabaseAdmin";

function extractStoragePath(publicUrl: string, bucket: string): string | null {
  if (!publicUrl) return null;

  if (publicUrl.startsWith(`${bucket}/`)) {
    return publicUrl.slice(bucket.length + 1);
  }

  try {
    const url = new URL(publicUrl);
    const marker = `/storage/v1/object/public/${bucket}/`;
    const idx = url.pathname.indexOf(marker);
    if (idx === -1) return null;
    return url.pathname.slice(idx + marker.length);
  } catch {
    return null;
  }
}

export async function DELETE(
  request: Request,
  context: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await context.params;

    if (!id) {
      return NextResponse.json({ error: "Ad ID is required" }, { status: 400 });
    }

    let userId: string | null = null;
    try {
      const body = await request.json();
      userId = body?.userId ?? null;
    } catch {
      userId = null;
    }

    const { data: phone, error: fetchError } = await supabaseAdmin
      .from("mobile_phones")
      .select("id, user_id, pictures, damage_report_pdf")
      .eq("id", id)
      .single();

    if (fetchError || !phone) {
      return NextResponse.json({ error: "Ad not found" }, { status: 404 });
    }

    if (userId && phone.user_id !== userId) {
      return NextResponse.json({ error: "Not authorized" }, { status: 403 });
    }

    const imagePaths = (phone.pictures || [])
      .map((url: string) => extractStoragePath(url, "phone-images"))
      .filter(Boolean) as string[];

    if (imagePaths.length > 0) {
      const { error: imageDeleteError } = await supabaseAdmin.storage
        .from("phone-images")
        .remove(imagePaths);

      if (imageDeleteError) {
        return NextResponse.json(
          { error: imageDeleteError.message },
          { status: 500 }
        );
      }
    }

    if (phone.damage_report_pdf) {
      const reportPath = extractStoragePath(
        phone.damage_report_pdf,
        "phone-reports"
      );
      if (reportPath) {
        const { error: reportDeleteError } = await supabaseAdmin.storage
          .from("phone-reports")
          .remove([reportPath]);

        if (reportDeleteError) {
          return NextResponse.json(
            { error: reportDeleteError.message },
            { status: 500 }
          );
        }
      }
    }

    const { error: deleteError } = await supabaseAdmin
      .from("mobile_phones")
      .delete()
      .eq("id", id);

    if (deleteError) {
      return NextResponse.json(
        { error: deleteError.message },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (err) {
    console.error(err);
    return NextResponse.json(
      { error: "Something went wrong" },
      { status: 500 }
    );
  }
}
