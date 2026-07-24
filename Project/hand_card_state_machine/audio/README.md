# Prerecorded Audio Files

Place these prerecorded files in this directory before running the system:

- `1.m4a` - Ask the student to show an ArUco card or speak their request.
- `2_item_request.m4a` - Confirm an item request.
- `3_teacher_request.m4a` - Confirm a teacher-assistance request.
- `4_teacher_loading.m4a` - Ask the teacher to place the item and press the button.
- `5_student_unloading.m4a` - Ask the student to collect the item and press the button.

The dispatcher uses this directory by default. Override it with
`ROBOT_AUDIO_DIRECTORY`; set `ROBOT_AUDIO_SERVER_URL` when the vehicle host
must reach the laptop through its Tailscale address.