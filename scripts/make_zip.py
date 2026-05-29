import shutil

shutil.make_archive(
    "submission_package/submission",
    "zip",
    "submission_package"
)

print("📦 submission.zip created")
