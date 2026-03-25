/* kernel.c - The YSU OS Shield Entry */
void _ysu_main() {
    // 0xB8000 is the address for the VGA text buffer in Protected Mode.
    // We cast it to a 'volatile char*' so the compiler doesn't optimize away our writes.
    volatile char* video_memory = (volatile char*) 0xB8000;
    
    char* msg = "YSU OS v1.7: THE CHISEL IS ALIVE.";
    
    // Each character on screen takes 2 bytes: [Character][Color/Attribute]
    // Color 0x0B is Light Cyan.
    for(int i = 0; msg[i] != '\0'; i++) {
        video_memory[i*2] = msg[i];     
        video_memory[i*2+1] = 0x0B;     
    }

    // Hang the CPU so it doesn't run off into random memory
    while(1);
}
