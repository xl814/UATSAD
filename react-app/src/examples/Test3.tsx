import { useRef, useState } from "react";
function VideoPlayer({src, isPlaying}){
    const ref = useRef(null);
    if (isPlaying){
        ref.current.play();
    } else {
        ref.current.pause();
    }
    return <video src={src} ref={ref} loop playsInline />
}

export default function Test3(){
    const [isPlaying, setIsPlaying] = useState(false);
    return (
        <>
            <button onClick={()=>setIsPlaying(!isPlaying)}>
                {isPlaying ? 'Pause': 'Play'}
            </button>
            <VideoPlayer
                isPlaying={isPlaying}
                src={"***"} />
        </>
    )
}