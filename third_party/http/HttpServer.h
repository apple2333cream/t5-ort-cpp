// 将HttpServer.h放到httplib.h文件的同级目录下

#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <typeinfo>
#include "httplib.h"

namespace httplib
{
	// 线程池(复制的httplib的线程池, 添加了动态增加线程数量的功能)
	class HttpServerThreadPool : public TaskQueue
	{
	public:

		explicit HttpServerThreadPool(size_t n) : shutdown_(false)
		{
			while (n)
			{
				threads_.emplace_back(worker(*this));
				n--;
			}
		}

		void addAThread()
		{
			threads_.emplace_back(worker(*this));
		}

		std::size_t getThreadNum()
		{
			return threads_.size();
		}

		HttpServerThreadPool(const ThreadPool&) = delete;
		~HttpServerThreadPool() override = default;

		void enqueue(std::function<void()> fn) override
		{
			{
				std::unique_lock<std::mutex> lock(mutex_);
				jobs_.push_back(std::move(fn));
			}

			cond_.notify_one();
		}

		void shutdown() override
		{
			// Stop all worker threads...
			{
				std::unique_lock<std::mutex> lock(mutex_);
				shutdown_ = true;
			}

			cond_.notify_all();

			// Join...
			for (auto& t : threads_)
			{
				t.join();
			}
		}

	private:
		struct worker
		{
			explicit worker(HttpServerThreadPool& pool) : pool_(pool) {}

			void operator()()
			{
				for (;;)
				{
					std::function<void()> fn;
					{
						std::unique_lock<std::mutex> lock(pool_.mutex_);

						pool_.cond_.wait(
							lock, [&]
							{ return !pool_.jobs_.empty() || pool_.shutdown_; });

						if (pool_.shutdown_ && pool_.jobs_.empty())
						{
							break;
						}

						fn = std::move(pool_.jobs_.front());
						pool_.jobs_.pop_front();
					}

					assert(true == static_cast<bool>(fn));
					fn();
				}
			}

			HttpServerThreadPool& pool_;
		};
		friend struct worker;

		std::vector<std::thread> threads_;
		std::list<std::function<void()>> jobs_;

		bool shutdown_;

		std::condition_variable cond_;
		std::mutex mutex_;
	};

	class BaseController;

	// HttpServer, 用于管理httplib::Server对象
	class HttpServer
	{
	private:
		std::string host = "";
		int port = -1;
		static httplib::HttpServerThreadPool* serverPool;
		httplib::Server* server = new httplib::Server();
		static std::atomic<int> poolUseNum;
		int socket_flags;
		void buildServerThreadPool();

		friend class BaseController;

	public:
		HttpServer(const HttpServer&) = delete;
		HttpServer() = delete;

		HttpServer(const std::string& _host, int _port, int _socket_flags = 0) : host(std::move(_host)), port(_port), socket_flags(_socket_flags) {}
		HttpServer(int _port, int _socket_flags = 0) : host(std::move("0.0.0.0")), port(_port), socket_flags(_socket_flags) {}

		httplib::Server* getHttplibServer();
		/*
			如果serverPool为null, 将为static serverPool 创建一个新的HttpThreadPool, 并在该线程池中监听
			如果serverPool不为null, 将直接使用static serverPool, 在线程池中执行httplib::Server的listen函数
		*/
		auto listenInThread();

		// 在本地监听httplib::Server的listen函数
		void listenInLocal();

		// 释放server指针,如果poolUseNum为0, 也将释放serverPool
		~HttpServer();
	};

	// BashController, 模仿java spring mvc的开发风格
	class BaseController
	{

	protected:
		// 必须重写bind方法, 在其中绑定具体的请求响应地址和请求响应方法, 否则抛出一个string异常
		virtual void bind()
		{
			throw std::string("must override ", __FUNCTION__);
		}
		httplib::Server* server = nullptr;

		// 绑定函数
		template<class Func, class T>
		auto BindController(Func&& func, T&& obj)
		{
			httplib::Server::Handler handler = std::bind(func, obj, std::placeholders::_1, std::placeholders::_2);
			return handler;
		}


	public:
		BaseController() = default;
		BaseController(BaseController&) = delete;

		void operator>>(httplib::Server* _server)
		{
			server = _server;
			// print("{} init to server", typeid(*this).name());
			this->bind();
		}

		void initToServer(httplib::Server* _server)
		{
			server = _server;
			// print("{} init to server", typeid(*this).name());
			this->bind();
		}

		void operator>>(httplib::HttpServer* _server)
		{

			server = _server->getHttplibServer();
			// print("{} init to server", typeid(*this).name());
			this->bind();
		}

		void initToServer(httplib::HttpServer* _server)
		{

			server = _server->getHttplibServer();
			// print("{} init to server", typeid(*this).name());
			this->bind();
		}



		~BaseController()
		{
			if (server != nullptr)
			{
				server = nullptr;
			}
			// print("destroy controller");
		}
	};

}

httplib::HttpServerThreadPool* httplib::HttpServer::serverPool = nullptr;
std::atomic_int httplib::HttpServer::poolUseNum(0);

void httplib::HttpServer::buildServerThreadPool()
{
	poolUseNum++;

	if (serverPool == nullptr)
	{

		serverPool = new httplib::HttpServerThreadPool{ (std::size_t)poolUseNum.load() };
	}

}

httplib::Server*
httplib::HttpServer::getHttplibServer()
{
	return server;
}

auto httplib::HttpServer::listenInThread()
{

	buildServerThreadPool();

	if (serverPool != nullptr)
	{
		serverPool->addAThread();
	}

	std::cout << "listen to " << port << std::endl;
	return serverPool->enqueue(std::bind(&httplib::Server::listen, server, host, port, socket_flags));
}

void httplib::HttpServer::listenInLocal()
{
	server->listen(host, port, socket_flags);
	// print("listen in port {}", port);
}

httplib::HttpServer::~HttpServer()
{

	if (this->server != nullptr)
	{
		delete (server);
	}

	poolUseNum--;
	if (poolUseNum == 0 && serverPool != nullptr)
	{
		delete (serverPool);
		serverPool = nullptr;
	}
}

