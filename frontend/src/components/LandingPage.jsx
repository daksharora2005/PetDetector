import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaShieldAlt, FaBrain, FaMobileAlt, FaArrowRight, FaCamera, FaPlay, FaTimes, FaUpload } from 'react-icons/fa';
import Typewriter from 'typewriter-effect';

const DemoModal = ({ isOpen, onClose }) => {
    const [videoSrc, setVideoSrc] = useState('/demo.mp4');
    const [videoError, setVideoError] = useState(false);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            setVideoSrc(url);
            setVideoError(false);
        }
    };

    const handleVideoError = () => {
        // If /demo.mp4 fails (e.g. 404), show upload UI
        setVideoError(true);
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        className="bg-gray-900 border border-white/10 rounded-2xl p-6 max-w-4xl w-full shadow-2xl relative"
                        onClick={e => e.stopPropagation()}
                    >
                        <button
                            onClick={onClose}
                            className="absolute -top-4 -right-4 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors z-10"
                        >
                            <FaTimes />
                        </button>

                        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                            <FaPlay className="text-brand-primary" /> Application Demo
                        </h2>

                        <div className="aspect-video bg-black rounded-xl overflow-hidden border border-white/10 flex flex-col items-center justify-center relative">
                            {!videoError ? (
                                <video
                                    src={videoSrc}
                                    controls
                                    autoPlay
                                    className="w-full h-full object-contain"
                                    onError={handleVideoError}
                                />
                            ) : (
                                <div className="text-center p-8">
                                    <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4 border border-dashed border-white/20">
                                        <FaUpload className="text-3xl text-gray-400" />
                                    </div>
                                    <p className="text-xl font-medium mb-2">Demo Video Not Found</p>
                                    <p className="text-gray-400 text-sm mb-6 max-w-md mx-auto">
                                        Please place a file named <code>demo.mp4</code> in the <code>frontend/public</code> folder.
                                        <br />Or upload one now:
                                    </p>
                                    <label className="btn-primary cursor-pointer inline-flex items-center gap-2">
                                        <input
                                            type="file"
                                            accept="video/*"
                                            className="hidden"
                                            onChange={handleFileChange}
                                        />
                                        Select Video File
                                    </label>
                                </div>
                            )}
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

const LandingPage = ({ onGetStarted }) => {
    const [showDemo, setShowDemo] = useState(false);

    return (
        <div className="min-h-screen bg-[#030014] text-white overflow-hidden relative selection:bg-brand-primary/30">
            {/* Background Effects */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:50px_50px]" />
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[400px] bg-brand-primary/20 blur-[120px] rounded-full opacity-50" />
                <div className="absolute bottom-0 right-0 w-[800px] h-[600px] bg-violet-600/10 blur-[120px] rounded-full opacity-30" />
            </div>

            <DemoModal isOpen={showDemo} onClose={() => setShowDemo(false)} />

            <div className="relative z-10 max-w-7xl mx-auto px-6 pt-32 pb-20">
                {/* Hero Section */}
                <div className="flex flex-col items-center text-center mb-24">
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-white/10 bg-white/5 backdrop-blur-md text-sm font-medium mb-8 hover:bg-white/10 transition-colors cursor-default"
                    >
                        <span className="flex h-2 w-2 rounded-full bg-brand-primary animate-pulse"></span>
                        Introducing PetGuard Pro
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.1 }}
                        className="text-6xl md:text-8xl font-bold tracking-tight mb-8 leading-tight"
                    >
                        Smart Security for <br />
                        <span className="bg-clip-text text-transparent bg-gradient-to-br from-white via-white to-white/50 pb-2 inline-block">
                            <Typewriter
                                options={{
                                    strings: ['Your Dog.', 'Your Cat.', 'Peace of Mind.'],
                                    autoStart: true,
                                    loop: true,
                                    delay: 50,
                                    deleteSpeed: 30,
                                }}
                            />
                        </span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="text-gray-400 text-lg md:text-xl max-w-2xl mb-10 leading-relaxed"
                    >
                        The world's first generalized AI pet door that learns to recognize <span className="text-white font-semibold">YOUR</span> specific pet.
                        Keep intruders out and let your furry friend in with generalized intelligence.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.3 }}
                        className="flex flex-wrap items-center justify-center gap-4"
                    >
                        <button onClick={onGetStarted} className="btn-primary flex items-center gap-3 text-lg px-8 py-4">
                            Get Started Now <FaArrowRight className="text-sm" />
                        </button>
                        <button onClick={() => setShowDemo(true)} className="btn-secondary flex items-center gap-3 text-lg px-8 py-4">
                            <FaPlay className="text-xs" /> Watch Demo
                        </button>
                    </motion.div>

                    {/* Hero Visual */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, rotateX: 20 }}
                        animate={{ opacity: 1, scale: 1, rotateX: 0 }}
                        transition={{ duration: 1, delay: 0.4 }}
                        className="mt-20 relative w-full max-w-4xl mx-auto perspective-1000"
                    >
                        <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-violet-600 rounded-2xl blur opacity-20 animate-pulse-soft" />
                        <div className="relative bg-[#0a0a0a] border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
                            <div className="flex items-center px-4 py-3 border-b border-white/10 bg-white/5">
                                <div className="flex gap-2">
                                    <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50" />
                                    <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
                                    <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50" />
                                </div>
                                <div className="mx-auto text-xs font-mono text-gray-500">live_feed_v2.0.exe</div>
                            </div>
                            <div className="aspect-video relative bg-black/50 flex flex-col items-center justify-center overflow-hidden group">
                                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=2874&auto=format&fit=crop')] bg-cover bg-center opacity-40 group-hover:opacity-60 transition-opacity duration-700" />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a0a] via-transparent to-transparent" />

                                {/* Scanning Effect */}
                                <div className="absolute inset-0 z-10 animate-scan pointer-events-none">
                                    <div className="w-full h-[2px] bg-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.8)]" />
                                </div>

                                <div className="relative z-20 bg-black/60 backdrop-blur-md px-6 py-3 rounded-full border border-white/10 flex items-center gap-3">
                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                    <span className="font-mono text-sm text-green-400">Target Identified: BO (99.8%)</span>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>

                {/* Features Layout (Bento Grid) */}
                <div className="mb-20">
                    <h2 className="text-3xl md:text-5xl font-bold text-center mb-16 bg-clip-text text-transparent bg-gradient-to-b from-white to-white/60">
                        Engineered for Peace of Mind
                    </h2>

                    <div className="grid md:grid-cols-3 gap-6 auto-rows-[300px]">
                        {/* Feature 1 (Span 2) */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="bento-card md:col-span-2 group"
                        >
                            <div className="absolute top-0 right-0 p-8 opacity-20 group-hover:opacity-40 transition-opacity">
                                <FaBrain className="text-9xl text-brand-primary rotate-12" />
                            </div>
                            <div className="relative z-10 h-full flex flex-col justify-end">
                                <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center text-2xl mb-6 backdrop-blur-sm border border-white/10">
                                    <FaBrain />
                                </div>
                                <h3 className="text-2xl font-bold mb-3">Self-Learning Neural Network</h3>
                                <p className="text-gray-400 max-w-md">
                                    Our proprietary AI doesn't just look for "a dog". It learns every angle, lighting condition, and quirk of YOUR specific pet in under 2 minutes.
                                </p>
                            </div>
                        </motion.div>

                        {/* Feature 2 */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 }}
                            className="bento-card group"
                        >
                            <div className="absolute -right-4 -top-4 w-32 h-32 bg-blue-500/20 rounded-full blur-[50px] group-hover:bg-blue-500/30 transition-colors" />
                            <div className="relative z-10 h-full flex flex-col justify-end">
                                <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center text-2xl mb-6 backdrop-blur-sm border border-white/10">
                                    <FaShieldAlt />
                                </div>
                                <h3 className="text-xl font-bold mb-3">Zero-Trust Security</h3>
                                <p className="text-gray-400">
                                    Raccoons, stray cats, and unknown animals are instantly locked out.
                                </p>
                            </div>
                        </motion.div>

                        {/* Feature 3 */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.2 }}
                            className="bento-card group"
                        >
                            <div className="absolute -left-4 -bottom-4 w-32 h-32 bg-violet-500/20 rounded-full blur-[50px] group-hover:bg-violet-500/30 transition-colors" />
                            <div className="relative z-10 h-full flex flex-col justify-end">
                                <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center text-2xl mb-6 backdrop-blur-sm border border-white/10">
                                    <FaMobileAlt />
                                </div>
                                <h3 className="text-xl font-bold mb-3">Real-time Command</h3>
                                <p className="text-gray-400">
                                    Get HD alerts on your phone. Lock or unlock your door from anywhere in the world.
                                </p>
                            </div>
                        </motion.div>

                        {/* Feature 4 (Span 2) */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.3 }}
                            className="bento-card md:col-span-2 relative overflow-hidden flex items-center"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-900/20 to-purple-900/20" />
                            <div className="relative z-10 p-8 text-center w-full">
                                <h3 className="text-2xl font-bold mb-4">Ready to upgrade your home?</h3>
                                <p className="text-gray-400 mb-8 max-w-lg mx-auto">Join thousands of happy pet owners who have reclaimed their peace of mind.</p>
                                <button onClick={onGetStarted} className="px-8 py-3 bg-white text-black font-bold rounded-full hover:bg-gray-200 transition-colors">
                                    Start Training Now
                                </button>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LandingPage;
