import { supabase } from '../lib/supabaseClient';

export async function getOrCreateConversation(
  userA: string,
  userB: string
) {
  const [u1, u2] = [userA, userB].sort()

  const { data: existing } = await supabase
    .from("conversations")
    .select("*")
    .eq("user1_id", u1)
    .eq("user2_id", u2)
    .single()

  if (existing) return existing

  const { data } = await supabase
    .from("conversations")
    .insert({ user1_id: u1, user2_id: u2 })
    .select()
    .single()

  return data
}
