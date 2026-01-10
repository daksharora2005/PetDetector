import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaRobot, FaTimes, FaPaperPlane } from 'react-icons/fa';
import { api } from '../api';

const ChatWidget = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', content: "Hi! I'm GuardBot. Ask me anything about PetGuard Pro!" }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            // Simulate reading delay
            await new Promise(r => setTimeout(r, 600));

            const res = await api.post('/chat', { message: userMsg.content });
            const botMsg = { role: 'assistant', content: res.data.response };

            setMessages(prev => [...prev, botMsg]);
        } catch (e) {
            setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I'm having trouble connecting to the server." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed bottom-6 right-6 z-50">
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.9 }}
                        className="mb-4 w-80 md:w-96 glass-panel overflow-hidden flex flex-col shadow-2xl border border-brand-primary/30"
                        style={{ height: '500px' }}
                    >
                        {/* Header */}
                        <div className="bg-brand-primary/90 p-4 flex justify-between items-center backdrop-blur-md">
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                                    <FaRobot className="text-white" />
                                </div>
                                <div>
                                    <h3 className="font-bold text-white">GuardBot AI</h3>
                                    <p className="text-xs text-blue-100 flex items-center gap-1">
                                        <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" /> Online
                                    </p>
                                </div>
                            </div>
                            <button onClick={() => setIsOpen(false)} className="text-white/80 hover:text-white">
                                <FaTimes />
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-900/50" ref={scrollRef}>
                            {messages.map((m, i) => (
                                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[80%] p-3 rounded-2xl text-sm ${m.role === 'user'
                                            ? 'bg-brand-primary text-white rounded-tr-none'
                                            : 'bg-white/10 text-gray-200 rounded-tl-none border border-white/5'
                                        }`}>
                                        {m.content}
                                    </div>
                                </div>
                            ))}
                            {loading && (
                                <div className="flex justify-start">
                                    <div className="bg-white/10 p-3 rounded-2xl rounded-tl-none flex gap-1 items-center">
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Input */}
                        <form onSubmit={handleSend} className="p-3 bg-slate-800/50 border-t border-white/10 flex gap-2">
                            <input
                                type="text"
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                placeholder="Ask about PetGuard..."
                                className="flex-1 bg-black/20 border border-white/10 rounded-full px-4 py-2 text-sm focus:outline-none focus:border-brand-primary/50"
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || loading}
                                className="w-10 h-10 bg-brand-primary rounded-full flex items-center justify-center text-white hover:bg-blue-600 disabled:opacity-50 transition-colors"
                            >
                                <FaPaperPlane className="text-sm" />
                            </button>
                        </form>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Toggle Button */}
            {!isOpen && (
                <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setIsOpen(true)}
                    className="w-14 h-14 bg-gradient-to-r from-blue-600 to-violet-600 rounded-full shadow-lg shadow-blue-500/40 flex items-center justify-center text-white text-2xl"
                >
                    <FaRobot />
                </motion.button>
            )}
        </div>
    );
};

export default ChatWidget;
