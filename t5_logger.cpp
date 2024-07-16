
#include "t5_logger.h"

std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;

int Logger::Init( const std::string& log_path, const int log_level, const bool display, const int max_file_size, const int max_rotate_file) {
	
	try
	{
		spdlog::level::level_enum spdlog_level = spdlog::level::info;
		switch (log_level)
		{
		case SPDLOG_LEVEL_TRACE:
			spdlog_level = spdlog::level::trace;
			break;
		case SPDLOG_LEVEL_DEBUG:
			spdlog_level = spdlog::level::debug;
			break;
		case SPDLOG_LEVEL_INFO:
			spdlog_level = spdlog::level::info;
			break;
		case SPDLOG_LEVEL_WARN:
			spdlog_level = spdlog::level::warn;
			break;
		case SPDLOG_LEVEL_ERROR:
			spdlog_level = spdlog::level::err;
			break;
		case SPDLOG_LEVEL_CRITICAL:
			spdlog_level = spdlog::level::critical;
			break;
		case SPDLOG_LEVEL_OFF:
			spdlog_level = spdlog::level::off;
			break;

		default:
			break;
		}

		auto max_size = 1024 * 1024 * max_file_size;
		auto max_files = max_rotate_file;

		if (display)
		{
			// std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink;
			// console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
			// auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_path, max_size, max_files);
			// std::initializer_list<spdlog::sink_ptr> sink_list = { console_sink, rotating_sink };
			// auto logger = std::make_shared<spdlog::logger>("T5", std::initializer_list<spdlog::sink_ptr>() = { console_sink, rotating_sink });
			// logger->flush_on(spdlog::level::debug);
			// logger_ = logger;

			auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();            
            // Initialize the logger with just the console sink
            auto logger = std::make_shared<spdlog::logger>("T5", console_sink);
            logger->flush_on(spdlog::level::debug);
            logger_ = logger;
		}
		else
		{
			auto rotating_logger = spdlog::rotating_logger_mt("T5", log_path, max_size, max_files);
			logger_ = rotating_logger;
		}
		logger_->set_level(spdlog_level);	
	}
	catch (const spdlog::spdlog_ex& ex)
	{
		std::cout << "Log init excption: {}" << ex.what() << std::endl;
		return false;
	}
	return true;
}

std::shared_ptr<spdlog::logger> Logger::getLogger() {
	return logger_;
}

