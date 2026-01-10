import React, { useRef, useState, useCallback } from 'react';
import Webcam from "react-webcam";
import { api } from '../api';
import { motion } from 'framer-motion';
import { FaCamera, FaStop, FaShieldAlt } from 'react-icons/fa';

const LiveWebcam = () => {
    const webcamRef = useRef(null);
    const [isLive, setIsLive] = useState(false);
    const [result, setResult] = useState(null);
    const [heatmap, setHeatmap] = useState(null);

    const capture = useCallback(async () => {
        if (!webcamRef.current) return;

        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;

        try {
            const res = await api.post('/predict-webcam', { image_base64: imageSrc });
            setResult(res.data);
            if (res.data.heatmap_base64) {
                setHeatmap(res.data.heatmap_base64);
            }
        } catch (e) {
            console.error(e);
        }
    }, [webcamRef]);

    // Polling effect
    React.useEffect(() => {
        let interval;
        if (isLive) {
            interval = setInterval(capture, 1000); // Check every second
        }
        return () => clearInterval(interval);
    }, [isLive, capture]);

    return (
        <div className="max-w-4xl mx-auto pt-24 px-6 text-center">
            <h2 className="text-3xl font-bold mb-8">Live Guard Mode</h2>

            <div className="glass-panel p-4 inline-block relative overflow-hidden">
                <div className="relative rounded-xl overflow-hidden bg-black aspect-video max-w-2xl mx-auto border-2 border-white/10">
                    {isLive ? (
                        <>
                            <Webcam
                                audio={false}
                                ref={webcamRef}
                                screenshotFormat="image/jpeg"
                                width={640}
                                height={480}
                                className="w-full h-full object-cover"
                            />
                            {/* Overlay UI */}
                            <div className="absolute top-4 right-4 flex gap-2">
                                <span className="bg-red-600 text-white text-xs px-2 py-1 rounded animate-pulse">LIVE</span>
                            </div>

                            {/* Heatmap Overlay (Optional Toggle could be added) */}
                            {heatmap && (
                                <img
                                    src={`data:image/jpeg;base64,${heatmap}`}
                                    className="absolute inset-0 w-full h-full opacity-40 mix-blend-screen pointer-events-none"
                                />
                            )}
                        </>
                    ) : (
                        <div className="flex items-center justify-center h-full flex-col">
                            <FaShieldAlt className="text-6xl text-gray-600 mb-4" />
                            <p className="text-gray-400">Camera Inactive</p>
                        </div>
                    )}
                </div>

                {/* Status Bar */}
                {isLive && result && (
                    <motion.div
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        className={`mt-4 p-4 rounded-lg flex justify-between items-center text-left border ${result.class_name === "Your Pet" ? "bg-green-500/10 border-green-500/30" : "bg-red-500/10 border-red-500/30" // Simple logic, assumes user class 1 is "Your Pet" roughly or just visual style
                            }`}
                    >
                        <div>
                            <p className="text-xs text-gray-400 uppercase tracking-widest">Detection</p>
                            <h3 className="text-2xl font-bold">{result.class_name || "Scanning..."}</h3>
                        </div>
                        <div className="text-right">
                            <p className="text-xs text-gray-400 uppercase">Confidence</p>
                            <p className="text-xl font-mono text-brand-primary">{((result.confidence || 0) * 100).toFixed(1)}%</p>
                        </div>
                    </motion.div>
                )}
            </div>

            <div className="mt-8">
                <button
                    onClick={() => setIsLive(!isLive)}
                    className={`fixed bottom-10 left-1/2 -translate-x-1/2 px-8 py-4 rounded-full font-bold text-lg shadow-2xl flex items-center gap-3 transition-all ${isLive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                        }`}
                >
                    {isLive ? <><FaStop /> Stop Guard</> : <><FaCamera /> Start Guard Mode</>}
                </button>
            </div>
        </div>
    );
};

export default LiveWebcam;
