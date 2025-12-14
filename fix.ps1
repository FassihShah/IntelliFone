$file = "c:\Users\DELL\OneDrive\Desktop\intellifone_fyp\fyp-web\app\add\page.tsx"
$content = Get-Content $file -Raw

# Fix 1: Add randomUUID import
$content = $content -replace "import Link from 'next/link';", "import { randomUUID } from 'crypto';" + [Environment]::NewLine + "import Link from 'next/link';"

# Fix 2: Update upload path
$old = @"
      for (const file of files) {
        const fileName = `$`{Date.now()}-$`{file.name}`;
        const { error } = await supabase.storage
          .from('phone-images')
          .upload(fileName, file);

        if (error) throw error;

        const { data: urlData } = supabase.storage
          .from('phone-images')
          .getPublicUrl(fileName);
"@

$new = @"
      for (const file of files) {
        const ext = file.name.split('.').pop();
        const filePath = `$`{user.id}/$`{randomUUID()}.$`{ext}`;
        const { error } = await supabase.storage
          .from('phone-images')
          .upload(filePath, file, { contentType: file.type });

        if (error) throw error;

        const { data: urlData } = supabase.storage
          .from('phone-images')
          .getPublicUrl(filePath);
"@

$content = $content -replace [regex]::Escape($old), $new

# Fix 3: Add user_id to form
$content = $content -replace "pictures: images,\n        },", "pictures: images,`n          user_id: user.id,`n        },"

Set-Content $file -Value $content
Write-Host "File updated successfully!"
