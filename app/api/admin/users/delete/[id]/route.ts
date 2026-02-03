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
  _request: Request,
  context: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await context.params;

    if (!id) {
      return NextResponse.json({ error: "User ID is required" }, { status: 400 });
    }

    // Fetch all ads by this user (to delete files too)
    const { data: ads, error: adsError } = await supabaseAdmin
      .from("mobile_phones")
      .select("id, pictures, damage_report_pdf")
      .eq("user_id", id);

    if (adsError) {
      return NextResponse.json({ error: adsError.message }, { status: 500 });
    }

    const imagePaths: string[] = [];
    const reportPaths: string[] = [];

    for (const ad of ads || []) {
      const adImagePaths = (ad.pictures || [])
        .map((url: string) => extractStoragePath(url, "phone-images"))
        .filter(Boolean) as string[];
      imagePaths.push(...adImagePaths);

      if (ad.damage_report_pdf) {
        const reportPath = extractStoragePath(
          ad.damage_report_pdf,
          "phone-reports"
        );
        if (reportPath) reportPaths.push(reportPath);
      }
    }

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

    if (reportPaths.length > 0) {
      const { error: reportDeleteError } = await supabaseAdmin.storage
        .from("phone-reports")
        .remove(reportPaths);

      if (reportDeleteError) {
        return NextResponse.json(
          { error: reportDeleteError.message },
          { status: 500 }
        );
      }
    }

    // Delete ads from DB
    const { error: deleteAdsError } = await supabaseAdmin
      .from("mobile_phones")
      .delete()
      .eq("user_id", id);

    if (deleteAdsError) {
      return NextResponse.json(
        { error: deleteAdsError.message },
        { status: 500 }
      );
    }

    // Delete profile row
    const { error: deleteProfileError } = await supabaseAdmin
      .from("profiles")
      .delete()
      .eq("id", id);

    if (deleteProfileError) {
      return NextResponse.json(
        { error: deleteProfileError.message },
        { status: 500 }
      );
    }

    // Best-effort: delete auth user
    try {
      await supabaseAdmin.auth.admin.deleteUser(id);
    } catch (err) {
      console.warn("Failed to delete auth user:", err);
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
