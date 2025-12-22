import React, { useState } from 'react';
import '../index.css';

// SVG Icons (Polished, Modern)
const ShieldIcon = () => (
    <svg className="w-10 h-10 text-sky-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
);

const BoltIcon = () => (
    <svg className="w-10 h-10 text-sky-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
);

const DocumentIcon = () => (
    <svg className="w-10 h-10 text-sky-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
);

const DownloadPage = () => {
    return (
        <div className="h-screen w-full overflow-y-auto bg-slate-50 text-slate-900 font-sans selection:bg-sky-100 selection:text-sky-900 scroll-smooth">

            {/* Navbar */}
            <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
                <div
                    className="max-w-7xl mx-auto h-16 flex items-center justify-between"
                    style={{ paddingLeft: '48px', paddingRight: '48px' }}
                >
                    <div className="flex items-center gap-2">
                        <img src="/Cellami_Template.png" alt="Cellami Logo" className="w-8 h-8 object-contain" />
                        <span className="text-lg font-bold tracking-tight text-slate-900">Cellami</span>
                    </div>
                </div>
            </header>

            {/* Hero Section */}
            <main
                className="relative z-10 px-6 flex flex-col items-center text-center"
                style={{ paddingTop: '120px', paddingBottom: '120px' }}
            >

                {/* Hero / Logo */}
                <div className="mb-12 relative animate-fade-in-up">
                    <div className="absolute inset-0 bg-sky-400/20 blur-[100px] rounded-full"></div>
                    <img
                        src="/Cellami_Template.png"
                        alt="Cellami AI Logo"
                        className="relative w-48 h-48 md:w-56 md:h-56 object-contain drop-shadow-xl mx-auto"
                    />
                </div>

                {/* Headline */}
                <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-8 text-slate-900 max-w-4xl leading-[1.1]">
                    Your Private AI Analyst <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-500 to-blue-600">
                        Built for Excel
                    </span>
                </h1>

                <p className="text-xl text-slate-600 mb-12 max-w-2xl mx-auto leading-relaxed font-medium">
                    Bring local LLMs directly into your spreadsheets. <br className="hidden md:block" />
                    No cloud. No subscriptions. 100% Private.
                </p>

                {/* CTAs */}
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full">
                    <a href="https://github.com/CameronKish/Cellami/releases/latest"
                        target="_blank" rel="noopener noreferrer"
                        className="min-w-[200px] px-8 py-4 rounded-full bg-sky-600 hover:bg-sky-500 text-white font-bold text-lg shadow-lg shadow-sky-200 transition-all transform hover:-translate-y-1">
                        Download Cellami
                    </a>
                    <a href="#how-it-works"
                        className="min-w-[200px] px-8 py-4 rounded-full bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 font-bold text-lg transition-all shadow-sm">
                        Installation Guide
                    </a>
                </div>
            </main>

            {/* Feature Grid (Strict Centering) */}
            <section
                className="px-6 bg-white border-y border-slate-100 flex flex-col items-center"
                style={{ paddingTop: '120px', paddingBottom: '120px' }}
            >
                <div className="max-w-5xl w-full">
                    <div className="text-center flex flex-col items-center" style={{ marginBottom: '72px' }}>
                        <h2 className="text-4xl font-bold text-slate-900 mb-6 tracking-tight">Why Go Local?</h2>
                        <p className="text-slate-500 text-lg max-w-xl mx-auto font-medium text-center">
                            Security, speed, and privacy without compromise.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-16 w-full place-items-center">
                        <FeatureCard
                            icon={<ShieldIcon />}
                            title="Private by Design"
                            desc="Your data never leaves your device. We use local embedding models and Ollama for complete security."
                        />
                        <FeatureCard
                            icon={<BoltIcon />}
                            title="Instant Results"
                            desc="By running on your own hardware, AI processes at the speed of thought. No internet required."
                        />
                        <FeatureCard
                            icon={<DocumentIcon />}
                            title="Chat with Data"
                            desc="Inform table queries with your personal knowledge base for AI enhanced decision making."
                        />
                    </div>
                </div>
            </section>

            {/* Setup Steps (Updated for Store-based flow) */}
            <section
                id="how-it-works"
                className="px-6 bg-slate-50 flex flex-col items-center"
                style={{ paddingTop: '120px', paddingBottom: '120px' }}
            >
                <div className="max-w-5xl w-full text-center">
                    <h2 className="text-4xl font-bold text-slate-900 tracking-tight" style={{ marginBottom: '72px' }}>
                        Simple Store Setup
                    </h2>

                    <div className="relative">
                        {/* Connecting Line - Manually Centered (Circle height 56px, so top-7 is 28px) */}
                        <div className="absolute top-7 left-[15%] right-[15%] h-0.5 bg-slate-200 hidden md:block opacity-60"></div>

                        <div className="grid md:grid-cols-3 gap-16 relative z-10 w-full place-items-center">
                            <Step number="1" title="Download Cellami" desc="Get the local AI app for MacOS or Windows from GitHub." />
                            <Step number="2" title="Add from Store" desc="Search for 'Cellami' in the Microsoft Excel Store and click Add." />
                            <Step number="3" title="Launch & Sync" desc="Open the Cellami app, then open the task pane in Excel." />
                        </div>
                    </div>
                </div>
            </section>

            {/* FAQ (Modern, Minimalist & Centered) */}
            <section
                className="px-6 bg-white border-t border-slate-100 flex flex-col items-center"
                style={{ paddingTop: '120px', paddingBottom: '120px' }}
            >
                <div className="max-w-2xl w-full text-center">
                    <h2 className="text-4xl font-bold mb-16 text-slate-900 tracking-tight">FAQ</h2>
                    <div className="divide-y divide-slate-200 w-full">
                        <AccordionItem q="Why do I need to download an 'App'?" a="Because Cellami runs LLMs locally on your hardware for 100% privacy, the desktop app handles the heavy processing that regular web-based add-ins can't." />
                        <AccordionItem q="Will this slow down my Excel?" a="No. Cellami runs as a separate background process, ensuring Excel remains lightweight and responsive." />
                        <AccordionItem q="Do I need a GPU?" a="It helps, but we support optimized models for modern CPUs (M1/M2/M3 or recent Intel/AMD) that run impressively fast." />
                        <AccordionItem q="Is it really free?" a="The engine is free and open source. You host the data and the models yourself." />
                    </div>
                </div>
            </section>

            <footer className="py-12 text-center text-slate-400 text-sm bg-slate-50 border-t border-slate-200">
                <p>&copy; {new Date().getFullYear()} Cellami AI. Built for privacy.</p>
            </footer>
        </div>
    );
};

// --- Sub Components ---

const FeatureCard = ({ icon, title, desc }) => (
    <div className="flex flex-col items-center text-center group w-full max-w-sm">
        <div className="mb-6 p-4 rounded-2xl bg-slate-50 group-hover:bg-sky-50 transition-colors duration-300 transform group-hover:-translate-y-1">
            {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-slate-900 tracking-tight">{title}</h3>
        <p className="text-slate-600 leading-relaxed text-sm lg:text-base">{desc}</p>
    </div>
);

const Step = ({ number, title, desc }) => (
    <div className="flex flex-col items-center text-center w-full max-w-xs">
        <div className="w-14 h-14 rounded-full bg-sky-500 text-white flex items-center justify-center font-bold text-2xl shadow-lg shadow-sky-200 mb-6 border-4 border-white relative z-10 transition-transform duration-300 group-hover:scale-110">
            {number}
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2 tracking-tight">{title}</h3>
        <p className="text-slate-500 text-sm font-medium leading-relaxed">{desc}</p>
    </div>
);

const AccordionItem = ({ q, a }) => {
    const [isOpen, setIsOpen] = useState(false);
    return (
        <div className="flex flex-col items-center" style={{ paddingTop: '36px', paddingBottom: '36px' }}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex flex-col items-center justify-center focus:outline-none group"
            >
                <div className="flex items-center justify-center gap-3">
                    <span className={`text-xl font-bold transition-colors ${isOpen ? 'text-sky-600' : 'text-slate-800'}`}>{q}</span>
                    <span className={`flex-shrink-0 transform transition-transform text-slate-300 ${isOpen ? 'rotate-180 text-sky-500' : ''}`}>
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
                        </svg>
                    </span>
                </div>
            </button>
            <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isOpen ? 'max-h-40 opacity-100 mt-6' : 'max-h-0 opacity-0'}`}>
                <p className="text-slate-600 text-base leading-relaxed font-medium max-w-xl mx-auto text-center">
                    {a}
                </p>
            </div>
        </div>
    );
};

export default DownloadPage;
