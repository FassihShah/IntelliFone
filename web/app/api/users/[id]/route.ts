import { NextResponse } from "next/server"
import { supabase } from "@/app/lib/supabaseClient"

console.log("route.ts loaded")

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> }
) {
  // ✅ unwrap params
  const { id } = await context.params

  console.log("Params:", { id })
  console.log("Request URL:", request.url)

  if (!id) {
    return NextResponse.json(
      { error: "User ID is required" },
      { status: 400 }
    )
  }

  const { data, error } = await supabase
    .from("profiles")
    .select("full_name, email")
    .eq("id", id)
    .single()

  if (error) {
    return NextResponse.json(
      { error: error.message },
      { status: 404 }
    )
  }

  return NextResponse.json(data)
}
