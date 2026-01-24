import React, { useState } from 'react';
import '../index.css';

// SVG Icons (Clean & Standard)
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
    const [showAbout, setShowAbout] = useState(false);
    return (
        <div className="h-screen w-full overflow-y-auto bg-slate-50 text-slate-900 font-sans selection:bg-sky-100 selection:text-sky-900 scroll-smooth">

            {/* Navbar */}
            <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-sm border-b border-slate-200">
                <div
                    className="max-w-5xl mx-auto h-16 flex items-center justify-between"
                    style={{ paddingLeft: '48px', paddingRight: '48px' }}
                >
                    <div
                        className="flex items-center gap-2 cursor-pointer group"
                        onClick={() => setShowAbout(true)}
                        title="About Cellami"
                    >
                        <img src="/Cellami_Template.png" alt="Cellami Logo" className="w-8 h-8 object-contain transition-transform group-hover:scale-110" />
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

            {/* App Showcase (Clean Carousel) */}
            <section className="px-6 pb-24 flex flex-col items-center">
                <div className="max-w-5xl w-full">
                    <AppShowcase />
                </div>
            </section>

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
                        Setup
                    </h2>

                    <div className="relative w-full max-w-5xl mx-auto">
                        {/* Connecting Line - Calculated for 4 columns: Centers are at 12.5% and 87.5% */}
                        <div className="absolute top-7 left-[12.5%] right-[12.5%] h-0.5 bg-slate-200 hidden md:block opacity-60"></div>

                        <div className="grid md:grid-cols-4 gap-4 relative z-10 w-full">
                            <Step number="1" title="Get the Add-in" desc="Add the manifest via Excel in the browser or sideload on desktop." />
                            <Step number="2" title="Download Cellami" desc="Get the companion desktop app for MacOS or Windows from GitHub." />
                            <Step number="3" title="Install Ollama" desc="Install the free AI engine from ollama.com to power your models." />
                            <Step number="4" title="Launch & Sync" desc="Run both apps, then open the Cellami task pane in Excel." />
                        </div>
                    </div>
                </div>
            </section>

            <section
                className="px-6 bg-white border-t border-slate-100 flex flex-col items-center"
                style={{ paddingTop: '120px', paddingBottom: '120px' }}
            >
                <div className="max-w-5xl w-full text-center">
                    <h2 className="text-4xl font-bold mb-16 text-slate-900 tracking-tight">FAQ</h2>
                    <div className="divide-y divide-slate-200 w-full">
                        <AccordionItem q="Why do I need to download two separate apps?" a="To provide 100% private AI, Cellami requires two local components: (1) Ollama, which acts as the 'brain' running the AI models on your hardware, and (2) the Cellami Desktop App, which securely bridges those models to Excel. This ensures your data never touches the cloud." />
                        <AccordionItem q="Will this slow down my Excel?" a="No. Cellami runs as a separate background process, ensuring Excel remains lightweight and responsive." />
                        <AccordionItem q="Do I need a GPU?" a="It helps, but we support optimized models for modern CPUs (M1/M2/M3 or recent Intel/AMD) that run impressively fast." />
                        <AccordionItem q="Is it really free?" a="The app is free and open source. You host the data and the models yourself." />
                    </div>
                </div>
            </section>

            <footer className="py-12 bg-slate-50 border-t border-slate-200 flex flex-col items-center text-center">
                <div className="max-w-5xl w-full px-6 flex flex-col items-center gap-6 text-slate-400 text-sm">
                    <p>&copy; {new Date().getFullYear()} Cellami. AI analysis, built privacy first.</p>
                    <div className="flex gap-8">
                        <a href="/privacy.html" target="_blank" rel="noopener noreferrer" className="hover:text-sky-600 transition-colors">Privacy</a>
                        <a href="/terms.html" target="_blank" rel="noopener noreferrer" className="hover:text-sky-600 transition-colors">Terms</a>
                        <a href="/support.html" target="_blank" rel="noopener noreferrer" className="hover:text-sky-600 transition-colors">Support</a>
                    </div>
                </div>
            </footer>

            {/* About Modal */}
            {showAbout && (
                <div
                    className="fixed inset-0 z-[3000] flex items-center justify-center bg-white/70 backdrop-blur-md animate-in fade-in duration-300"
                    onClick={() => setShowAbout(false)}
                >
                    <div
                        className="bg-white p-8 max-w-[320px] w-full relative overflow-hidden text-center animate-in zoom-in-95 duration-200 rounded-3xl shadow-2xl border border-slate-100"
                        onClick={e => e.stopPropagation()}
                    >
                        <div className="flex justify-center mb-6 relative">
                            <img src="/Cellami_Template.png" alt="Cellami Logo" className="w-16 h-16 drop-shadow-md relative" />
                        </div>

                        <h2 className="text-2xl font-bold text-slate-800 mb-1" style={{ letterSpacing: '-0.02em' }}>Cellami</h2>
                        <p className="text-sky-600 font-mono text-xs mb-6 font-bold tracking-wider">VERSION 1.2.0</p>

                        <p className="text-slate-600 text-sm leading-relaxed mb-8">
                            The intelligent companion for Excel.<br />
                            Local. Private. Secure.
                        </p>

                        <div className="flex flex-col gap-3">
                            <button onClick={() => setShowAbout(false)} className="px-6 py-2.5 bg-sky-600 hover:bg-sky-500 text-white font-bold rounded-xl transition-all">Close</button>

                            {/* Legal Links (Parity with Main App) */}
                            <div className="flex justify-center gap-4 mt-2">
                                <a href="/privacy.html" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-sky-600 text-[10px] transition-colors">Privacy</a>
                                <span className="text-slate-300">·</span>
                                <a href="/terms.html" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-sky-600 text-[10px] transition-colors">Terms</a>
                                <span className="text-slate-300">·</span>
                                <a href="/support.html" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-sky-600 text-[10px] transition-colors">Support</a>
                            </div>

                            <p className="text-slate-400 text-[9px] uppercase tracking-[0.2em] mt-2 font-bold opacity-80">Empowering Data Analysts</p>
                        </div>
                    </div>
                </div>
            )}
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
    <div className="flex flex-col items-center text-center w-full max-w-xs mx-auto">
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

const AppShowcase = () => {
    const [activeTab, setActiveTab] = useState(0);

    const tabs = [
        {
            id: 'docs',
            title: '1. Knowledge Base',
            desc: 'Upload your own documents (PDFs, Word, etc.) and ask questions based on them. Cellami uses your local files to provide accurate, context-aware answers.',
            image: '/screenshots/docs-hd.png'
        },
        {
            id: 'cell',
            title: '2. Query Cells',
            desc: 'Select any cell in your spreadsheet and ask Cellami to process it. Perfect for quick analysis, summarization, or extraction tasks on specific data points.',
            image: '/screenshots/cell-hd.png'
        },
        {
            id: 'table',
            title: '3. Query Tables',
            desc: 'Analyze entire tables at once. Select a range of data, and Cellami will analyze the full table or individual entries sequentially to provide comprehensive insights.',
            image: '/screenshots/table-hd.png'
        },
        {
            id: 'audit',
            title: '4. Audit Answers',
            desc: 'Trace back every answer to its source. The Audit tab shows you exactly which document chunks were used to generate a response, ensuring transparency and trust.',
            image: '/screenshots/audit-hd.png'
        },
        {
            id: 'chat',
            title: '5. Chat with Tables',
            desc: 'Have a free-form conversation with your data. The Chat tab allows for open-ended questions and follow-ups, maintaining context throughout your session.',
            image: '/screenshots/chat-hd.png'
        }
    ];

    const nextTab = () => {
        setActiveTab((prev) => (prev + 1) % tabs.length);
    };

    const prevTab = () => {
        setActiveTab((prev) => (prev - 1 + tabs.length) % tabs.length);
    };

    return (
        <div className="w-full max-w-5xl mx-auto flex flex-col items-center gap-10">

            {/* Carousel Container - Clean & Simple */}
            <div className="relative w-full group">

                {/* Image Frame (Standard) */}
                <div
                    className="relative w-full bg-slate-900 rounded-2xl shadow-xl overflow-hidden border border-slate-200"
                    style={{ aspectRatio: '16/9' }}
                >
                    {tabs.map((tab, index) => (
                        <img
                            key={tab.id}
                            src={tab.image}
                            alt={tab.title}
                            className={`absolute inset-0 w-full h-full object-contain bg-slate-900 transition-opacity duration-300 ${activeTab === index ? 'opacity-100' : 'opacity-0'
                                }`}
                        />
                    ))}
                </div>

                {/* Navigation Buttons (Standard) */}
                <button
                    onClick={prevTab}
                    className="absolute top-1/2 -left-2 md:-left-12 -translate-y-1/2 p-3 rounded-full bg-white shadow-md text-slate-700 hover:text-sky-600 hover:scale-110 transition-all z-10 border border-slate-100"
                    aria-label="Previous screenshot"
                >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                </button>

                <button
                    onClick={nextTab}
                    className="absolute top-1/2 -right-2 md:-right-12 -translate-y-1/2 p-3 rounded-full bg-white shadow-md text-slate-700 hover:text-sky-600 hover:scale-110 transition-all z-10 border border-slate-100"
                    aria-label="Next screenshot"
                >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>

            {/* Text Below */}
            <div className="text-center max-w-2xl animate-in fade-in slide-in-from-bottom-4 duration-300" key={activeTab}>
                <h3 className="text-2xl font-bold text-slate-900 mb-3 tracking-tight">
                    {tabs[activeTab].title}
                </h3>
                <p className="text-slate-500 text-lg leading-relaxed font-medium">
                    {tabs[activeTab].desc}
                </p>

                {/* Dots Indicator */}
                <div className="flex justify-center gap-3 mt-8">
                    {tabs.map((_, index) => (
                        <button
                            key={index}
                            onClick={() => setActiveTab(index)}
                            className={`h-2 rounded-full transition-all duration-300 ${activeTab === index ? 'w-8 bg-sky-500' : 'w-2 bg-slate-300 hover:bg-slate-400'
                                }`}
                            aria-label={`Go to slide ${index + 1}`}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default DownloadPage;
