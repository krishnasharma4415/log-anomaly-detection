import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';

export default function Layout() {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 transition-colors duration-200">
      <Sidebar />
      <div className="ml-64">
        <TopBar />
        <main className="p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
